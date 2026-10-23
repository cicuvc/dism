// Opt-in host compiler and benchmark binding. Production decoder is untouched.
#include "summary_program.h"
#include "../decode_rebuild.hpp"
#include "../prefix_type.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cmath>
#include <map>
#include <numeric>

extern "C" void launch_parallel_rebuild(const void*,const void*,const int*,const float*,DismPrefix*,DismPrefix*,float*,float*,float*,int,int,int,int,int,int,float,bool,cudaStream_t);

template<class T> torch::Tensor tensor(const std::vector<T>& x) {
    auto t=torch::empty({int64_t(x.size())},torch::TensorOptions().dtype(std::is_same_v<T,int>?torch::kInt32:torch::kFloat32));
    if(!x.empty())std::memcpy(t.data_ptr(),x.data(),x.size()*sizeof(T));return t;
}
class SummaryProgram {
    int h,n,r,d,block,tile,tiles,nn=0,nb=0,nc=0,nf=0,nm=0,jobs=0,qcap=0,maxnodes=0;
    std::vector<dism_decode::Planner> planners;
    std::vector<torch::Tensor> cpu,device,old_topo,old_coef;
    std::vector<int> levels,counts,offsets;
    torch::Tensor prefix,carry,tail,out,queue,ready,control,initial;
    torch::Tensor bp,bt,bping,bpong;
    torch::Device owner=torch::Device(torch::kCPU);
    cudaStream_t stream=nullptr;
    std::map<std::string,int64_t> metrics;
  public:
    SummaryProgram(torch::Tensor labels,std::vector<double> tau,int r_,int d_,int block_,int tile_)
        : h(labels.dim()==2?labels.size(0):0),n(labels.dim()==2?labels.size(1):0),r(r_),d(d_),block(block_),tile(tile_) {
        TORCH_CHECK(labels.device().is_cpu() && labels.scalar_type()==torch::kInt32 && labels.is_contiguous() && h>0 && n>0,"CPU int32 [H,N] keys required");
        TORCH_CHECK((r==16||r==32)&&(d==32||d==64)&&int(tau.size())==h,"R16/32 DV32/64 and H taus required");
        TORCH_CHECK(tile==32||tile==64||tile==128||tile==256,"tile must be 32/64/128/256");
        TORCH_CHECK(n<=(1<<20) && h<=128,"probe bounds: N<=2^20, H<=128");
        if(!block)block=int(std::ceil(std::sqrt(2.*n))); // same block width across all heads
        TORCH_CHECK(block>0&&block<=n*2,"block size bounds");tiles=(r*d+tile-1)/tile;
        std::vector<int> hd,node,co{0},cn,ao{0},anc,init;
        std::vector<float> w,rho,sc,asc;
        int64_t work=0,span=0,active=0,max_chain=0,max_anc=0,max_blocks=0,baseline_launches=0;
        for(int hi=0;hi<h;++hi) {
            planners.emplace_back(r,128,4*r,5*r,tau[hi]);auto &p=planners.back();
            const int *k=labels.data_ptr<int>()+int64_t(hi)*n;
            p.keys.assign(k,k+n);p.rebuild();
            dism_decode::RebuildWorkspace ws;ws.prepare(p);
            int count=p.materialized.size()+p.samples.size(),nodes=p.nodes.size();
            offsets.push_back(nm);counts.push_back(count);levels.push_back(ws.levels);
            maxnodes=std::max(maxnodes,nodes);
            // Exact production baseline metadata and CPU coefficient generation.
            std::vector<int> ot((8+ws.levels)*nodes);std::vector<float> oc((1+ws.levels)*nodes);
            ws.pack_summary(p,ot.data(),oc.data());old_topo.push_back(tensor(ot));old_coef.push_back(tensor(oc));
            if(!count) {hd.insert(hd.end(),{nn,0,nb,0,nc,0,nm,0,0,0});continue;}
            ++active;jobs+=tiles;
            baseline_launches+=((r*d+31)/32)*(3+ws.levels+(ws.levels>0));
            int blocks=(nodes+block-1)/block,chain_base=nc;
            std::vector<int> size(nodes,1),heavy(nodes,-1),which(nodes,-1),parent,parent_node;
            std::vector<double> node_scale(nodes,0.);
            for(auto it=p.order.rbegin();it!=p.order.rend();++it) {
                int u=*it;for(int v:p.children[u]) {size[u]+=size[v];if(heavy[u]<0||size[v]>size[heavy[u]])heavy[u]=v;}
            }
            node.resize((nn+nodes)*7);w.resize(nn+nodes);rho.resize(nn+nodes);sc.resize(nn+nodes);
            for(int u=0;u<nodes;++u) {
                int at=7*(nn+u);node[at]=p.nodes[u].position;node[at+1]=nn+p.order[u];
                node[at+2]=ws.inverse[u];node[at+3]=ws.end[u];
                node[at+4]=p.mat_index[u]<0?-1:nm+p.mat_index[u];
                node[at+5]=p.sample_index[u]<0?-1:nm+int(p.materialized.size())+p.sample_index[u];
                node[at+6]=-1;
                w[nn+u]=p.edge_weight[u];rho[nn+u]=p.edge_decay[u];
            }
            // Heavy paths keep the number of light edges logarithmic. Cut every
            // path into <=sqrt(N) segments; no recursive parallel scan hierarchy.
            int local_nodes=0,local_max=0;
            if(p.samples.empty()) {
                // With no sampled prefixes, only materialized subtree outputs
                // are live. Compile batched independent range queries instead
                // of dispatching millions of one-node chain instructions.
                for(size_t i=0;i<p.materialized.size();) {
                    ++nc;parent.push_back(-1);parent_node.push_back(-1);int length=0;
                    while(i<p.materialized.size() && length<block) {
                        int u=p.materialized[i++];cn.push_back(nn+u);sc[nn+u]=0.;++length;
                    }
                    co.push_back(cn.size());local_nodes+=length;local_max=std::max(local_max,length);
                }
            } else {
            std::vector<int> roots{0};
            while(!roots.empty()) {
                int u=roots.back();roots.pop_back();
                while(u>=0) {
                    int id=nc++,par=p.nodes[u].link;parent.push_back(par<0?-1:which[par]);parent_node.push_back(par);
                    double decay=1.;int length=0;
                    while(u>=0 && length<block) {
                        which[u]=id;cn.push_back(nn+u);decay*=p.edge_decay[u];sc[nn+u]=decay;node_scale[u]=decay;++length;
                        for(int v:p.children[u])if(v!=heavy[u])roots.push_back(v);
                        u=heavy[u];
                    }
                    co.push_back(cn.size());max_chain=std::max(max_chain,int64_t(length));
                }
            }
                local_nodes=nodes;local_max=std::min(block,nodes);
            }
            max_chain=std::max(max_chain,int64_t(local_max));
            // A light child may attach inside its parent's segment. Keep the
            // local affine value at that exact node, not just the segment tail.
            for(int u:parent_node)if(u>=0 && node[7*(nn+u)+6]<0)node[7*(nn+u)+6]=nf++;
            int longest=0;
            for(int id=chain_base;id<nc;++id) {
                double scale=1.;int current=id,a=parent[id-chain_base],length=0;
                if(!p.samples.empty())while(a>=0) {
                    int u=parent_node[current-chain_base];
                    anc.push_back(node[7*(nn+u)+6]);asc.push_back(scale);scale*=node_scale[u];
                    current=a;a=parent[a-chain_base];++length;
                }
                ao.push_back(anc.size());longest=std::max(longest,length);
            }
            int chains=nc-chain_base;
            hd.insert(hd.end(),{nn,nodes,nb,blocks,chain_base,chains,nm,count,int(!p.samples.empty()),0});
            for(int t=0;t<tiles;++t)for(int b=0;b<blocks;++b)init.insert(init.end(),{0,hi,t,b});
            int64_t extra=int64_t(tiles)*(blocks+1+int64_t(chains)*(p.samples.empty()?1:2));
            TORCH_CHECK(extra+qcap<=INT_MAX/4,"program task address exceeds int32");
            qcap+=int(extra);
            int64_t ancs=ao.back()-ao[chain_base];
            work+=nodes+blocks+local_nodes+(p.samples.empty()?0:local_nodes+ancs);
            span=std::max(span,int64_t(std::min(block,nodes)+blocks+local_max+(p.samples.empty()?0:longest+local_max)));
            max_anc=std::max(max_anc,int64_t(longest));max_blocks=std::max(max_blocks,int64_t(blocks));
            nn+=nodes;nb+=blocks;nm+=count;
        }
        cpu={tensor(hd),tensor(node),tensor(co),tensor(cn),tensor(ao),tensor(anc),tensor(w),tensor(rho),tensor(sc),tensor(asc),tensor(init)};
        metrics={{"heads",h},{"active_heads",active},{"tokens",n},{"nodes",nn},{"matrices",nm},{"block_size",block},{"tile",tile},{"tiles_per_head",tiles},
                 {"euler_blocks",nb},{"chain_blocks",nc},{"boundary_nodes",nf},{"max_chain_nodes",max_chain},{"max_ancestor_blocks",max_anc},
                 {"euler_tasks",int64_t(nb)*tiles},{"local_chain_tasks",int64_t(nc)*tiles},{"total_tasks",qcap},
                 {"baseline_launches",baseline_launches},{"program_launches",jobs?2:0},
                 {"node_work_per_coordinate",work},{"estimated_span_node_iterations",span},{"max_euler_blocks_per_head",max_blocks},
                 {"scratch_bytes",int64_t(nn+nb+nf)*r*d*4+int64_t(qcap)*20+(3+3*h*tiles)*4}};
        int64_t bytes=0;for(auto &x:cpu)bytes+=x.nbytes();metrics["program_bytes"]=bytes;
    }
    std::map<std::string,int64_t> stats() const{return metrics;}
    void prepare(torch::Tensor prototype) {
        TORCH_CHECK(prototype.is_cuda(),"CUDA prototype required");owner=prototype.device();c10::cuda::CUDAGuard guard(owner);stream=at::cuda::getCurrentCUDAStream();
        device.clear();for(auto &x:cpu)device.push_back(x.to(owner));
        for(auto &x:old_topo)x=x.to(owner);for(auto &x:old_coef)x=x.to(owner);
        auto f=prototype.options().dtype(torch::kFloat32),i=prototype.options().dtype(torch::kInt32);
        prefix=torch::empty({int64_t(nn),r*d},f);carry=torch::empty({int64_t(nb),r*d},f);tail=torch::empty({int64_t(nf),r*d},f);
        out=torch::empty({nm,r*d},f);queue=torch::empty({qcap,4},i);ready=torch::empty({qcap},i);control=torch::empty({3+3*h*tiles},i);initial=device[10];
        bp=torch::empty({maxnodes*32},f);bt=torch::empty({((maxnodes+255)/256)*32},f);bping=torch::empty_like(bp);bpong=torch::empty_like(bp);
    }
    void validate(torch::Tensor k,torch::Tensor v) {
        TORCH_CHECK(k.is_cuda()&&k.device()==owner&&v.device()==owner&&k.is_contiguous()&&v.is_contiguous(),"prepared device/contiguous payload required");
        TORCH_CHECK(k.dim()==3&&v.dim()==3&&k.size(0)==h&&v.size(0)==h&&k.size(1)>=n&&k.size(1)==v.size(1)&&k.size(2)==r&&v.size(2)==d,"payload [H,capacity,R/DV]");
        TORCH_CHECK(k.scalar_type()==v.scalar_type()&&(k.scalar_type()==torch::kBFloat16||k.scalar_type()==torch::kFloat32),"FP32/BF16 payload");
        TORCH_CHECK(at::cuda::getCurrentCUDAStream()==stream,"probe stream mismatch");
    }
    torch::Tensor run(torch::Tensor k,torch::Tensor v,int workers) {
        c10::cuda::CUDAGuard guard(k.device());validate(k,v);TORCH_CHECK(workers>0&&workers<=1024,"workers in [1,1024]");
        SummaryProgramArgs a{k.data_ptr(),v.data_ptr(),device[0].data_ptr<int>(),device[1].data_ptr<int>(),device[2].data_ptr<int>(),device[3].data_ptr<int>(),
            device[4].data_ptr<int>(),device[5].data_ptr<int>(),device[6].data_ptr<float>(),device[7].data_ptr<float>(),device[8].data_ptr<float>(),device[9].data_ptr<float>(),
            prefix.data_ptr<float>(),carry.data_ptr<float>(),tail.data_ptr<float>(),out.data_ptr<float>(),queue.data_ptr<int>(),ready.data_ptr<int>(),control.data_ptr<int>(),
            h,int(k.size(1)),r,d,tile,tiles,block,qcap,int(initial.numel()/4),jobs};
        launch_summary_program(a,initial.data_ptr<int>(),k.scalar_type()==torch::kBFloat16,workers,stream);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
    }
    torch::Tensor baseline(torch::Tensor k,torch::Tensor v) {
        c10::cuda::CUDAGuard guard(k.device());validate(k,v);
        for(int hi=0;hi<h;++hi)if(counts[hi]) {
            int nodes=planners[hi].nodes.size();
            const char *kp=(const char*)k.data_ptr()+int64_t(hi)*k.size(1)*r*k.element_size();
            const char *vp=(const char*)v.data_ptr()+int64_t(hi)*v.size(1)*d*v.element_size();
            for(int start=0;start<r*d;start+=32)launch_parallel_rebuild(kp,vp,old_topo[hi].data_ptr<int>(),old_coef[hi].data_ptr<float>(),
                (DismPrefix*)bp.data_ptr<float>(),(DismPrefix*)bt.data_ptr<float>(),bping.data_ptr<float>(),bpong.data_ptr<float>(),out.data_ptr<float>()+int64_t(offsets[hi])*r*d,
                nodes,r,d,start,std::min(32,r*d-start),levels[hi],planners[hi].tau,k.scalar_type()==torch::kBFloat16,stream);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
    }
    torch::Tensor reference(torch::Tensor k,torch::Tensor v) {
        TORCH_CHECK(k.device().is_cpu()&&v.device().is_cpu()&&k.scalar_type()==torch::kFloat32&&v.scalar_type()==torch::kFloat32&&k.is_contiguous()&&v.is_contiguous(),"CPU FP32 contiguous vectors required");
        TORCH_CHECK(k.sizes()==torch::IntArrayRef({h,n,r})&&v.sizes()==torch::IntArrayRef({h,n,d}),"reference shape");
        auto result=torch::empty({nm,r*d},torch::TensorOptions().dtype(torch::kFloat64));auto dest=result.data_ptr<double>();
        const float *key=k.data_ptr<float>(),*val=v.data_ptr<float>();std::vector<double> scratch;
        for(int hi=0;hi<h;++hi)if(counts[hi]) {
            const auto &p=planners[hi];int nodes=p.nodes.size(),width=r*d;scratch.assign(int64_t(nodes)*width,0.);
            for(int u=0;u<nodes;++u)if(p.nodes[u].position>=0) {
                int pos=p.nodes[u].position;
                for(int c=0;c<width;++c)scratch[int64_t(u)*width+c]=double(key[(int64_t(hi)*n+pos)*r+c/d])*double(val[(int64_t(hi)*n+pos)*d+c%d]);
            }
            for(auto it=p.order.rbegin();it!=p.order.rend();++it)if(*it) {
                int u=*it,par=p.nodes[u].link;
                for(int c=0;c<width;++c)scratch[int64_t(par)*width+c]+=scratch[int64_t(u)*width+c];
            }
            for(int u:p.order)for(int c=0;c<width;++c) {
                double sub=scratch[int64_t(u)*width+c];
                if(p.mat_index[u]>=0)dest[int64_t(offsets[hi]+p.mat_index[u])*width+c]=sub;
                double value=u?p.edge_weight[u]*sub+p.edge_decay[u]*scratch[int64_t(p.nodes[u].link)*width+c]:0.;
                scratch[int64_t(u)*width+c]=value;
                if(p.sample_index[u]>=0)dest[int64_t(offsets[hi]+p.materialized.size()+p.sample_index[u])*width+c]=value;
            }
        }
        return result;
    }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    pybind11::class_<SummaryProgram>(m,"SummaryProgram")
        .def(pybind11::init<torch::Tensor,std::vector<double>,int,int,int,int>())
        .def("stats",&SummaryProgram::stats).def("prepare",&SummaryProgram::prepare)
        .def("run",&SummaryProgram::run,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("baseline",&SummaryProgram::baseline,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("reference",&SummaryProgram::reference,pybind11::call_guard<pybind11::gil_scoped_release>());
}
