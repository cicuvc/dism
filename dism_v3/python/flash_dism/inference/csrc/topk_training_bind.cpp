#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "decoupled_readout_direct.hpp"
#include <chrono>
#include <climits>
namespace py=pybind11;
using namespace decoupled;
using I=py::array_t<int,py::array::c_style>;
using D=py::array_t<double,py::array::c_style>;
py::dict prepare_direct(std::shared_ptr<dism_prefill::PlanningWorkers>pool,I labels,D tau,I candidates,
                       int topk,bool gradients,int bk,int bq,int threshold,py::object allocator,
                       py::array_t<uint8_t,py::array::c_style> hard,
                       py::array_t<int64_t,py::array::c_style> documents) {
    if(labels.ndim()!=4||labels.shape(0)!=2||tau.ndim()!=1||tau.shape(0)!=labels.shape(2))throw std::invalid_argument("labels [2,B,H,N], tau [H]");
    if(labels.shape(1)>INT_MAX||labels.shape(2)>INT_MAX||labels.shape(3)>INT_MAX)throw std::invalid_argument("int32 dimensions required");
    int b=labels.shape(1),h=labels.shape(2),n=labels.shape(3);
    if(b<1||h<1||n<1||(topk!=1&&topk!=2&&topk!=4)||int64_t(b)*h>INT_MAX||int64_t(b)*h*n>INT_MAX/topk||threshold< -1)throw std::invalid_argument("shape/virtual index overflow or invalid threshold");
    if(topk>1&&(candidates.ndim()!=4||candidates.shape(0)!=b||candidates.shape(1)!=h||candidates.shape(2)!=n||candidates.shape(3)!=topk))throw std::invalid_argument("candidates [B,H,N,K]");
    if((bk!=16&&bk!=32&&bk!=64)||(bq!=16&&bq!=32&&bq!=64))throw std::invalid_argument("tile sizes 16/32/64 required");
    int width=gradients?64:16;
    auto ptr=labels.data();auto tp=tau.data();auto cp=topk>1?candidates.data():nullptr;
    if(hard.ndim()!=2||hard.shape(0)!=b||hard.shape(1)!=h||documents.ndim()!=2||documents.shape(1)!=3)
        throw std::invalid_argument("hard [B,H], documents [docs,3] required");
    std::vector<topk_prefill::TrainingJob>jobs;
    auto hp=hard.data();auto dp=documents.data();std::vector<int>ends(b,0);
    for(int64_t j=0;j<documents.shape(0);++j) {
        auto row=dp[j*3],start=dp[j*3+1],end=dp[j*3+2];
        if(row<0||row>=b||start!=ends[row]||end<=start||end>n||start%256||end%256)
            throw std::invalid_argument("documents must partition aligned training rows");
        ends[row]=end;
        for(int head=0;head<h;++head)if(hp[row*h+head])jobs.push_back({int(row),head,int(start),int(end),tp[head]});
    }
    for(auto end:ends)if(end!=n)throw std::invalid_argument("incomplete document coverage");
    for(int head=0;head<h;++head)if(!std::isfinite(tp[head]))throw std::invalid_argument("finite tau required");
    std::vector<topk_prefill::Program>programs(jobs.size());
    std::vector<DirectSize>sizes(jobs.size()),bases(jobs.size()+1);
    auto begin=std::chrono::steady_clock::now();
    {
        py::gil_scoped_release release;
        pool->run(programs.size(),[&](size_t i) {
            static thread_local topk_prefill::Builder builder;
            builder.cache_forward_weights=true;builder.lazy_gradient_weights=true;
            const auto&job=jobs[i];
            int64_t at=(int64_t(job.batch)*h+job.head)*n+job.start;
            builder.build(ptr+at,ptr+int64_t(b)*h*n+at,nullptr,job.end-job.start,job.tau,false,gradients,cp?cp+at*topk:nullptr,topk);
            programs[i]=builder.finish();sizes[i]=direct_count(programs[i],width,bk,bq,threshold,gradients);
        });
        for(size_t i=0;i<sizes.size();++i){bases[i+1]=bases[i];bases[i+1]+=sizes[i];}
    }
    auto built=std::chrono::steady_clock::now();auto total=bases.back();
    DirectLayout layout;layout.layout(total,width,bk,bq);
    if(layout.bytes>size_t(PY_SSIZE_T_MAX))throw std::overflow_error("Python arena size overflow");
    py::array_t<uint8_t>storage;
    if(allocator.is_none())storage=py::array_t<uint8_t>(layout.bytes);
    else {
        auto allocation=allocator(layout.bytes).cast<py::array>();
        if(allocation.ndim()!=1||allocation.size()!=py::ssize_t(layout.bytes)||!allocation.writeable()||
           !(allocation.flags()&py::array::c_style)||!allocation.dtype().is(py::dtype::of<uint8_t>())||
           reinterpret_cast<uintptr_t>(allocation.data())%8)
            throw std::invalid_argument("allocator must return an aligned writable contiguous uint8 arena of the requested size");
        storage=py::reinterpret_borrow<py::array_t<uint8_t>>(allocation);
    }
    layout.bytes=0;layout.data=storage.mutable_data();
    auto view=layout.layout(total,width,bk,bq);
    view.offsets[0]=view.qoffsets[0]=0;for(auto&a:view.old)a.offsets[0]=0;
    auto allocated=std::chrono::steady_clock::now();
    {
        py::gil_scoped_release release;
        pool->run(programs.size(),[&](size_t i) {
            const auto&job=jobs[i];
            direct_pack(programs[i],job,n,h,width,bk,bq,threshold,gradients,bases[i],view);
            // Reclaim the event arrays on workers, not serially on the caller.
            programs[i]=topk_prefill::Program{};
        },true);
    }
    auto done=std::chrono::steady_clock::now();
    auto milliseconds=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
    py::dict result,stats,split;py::list buckets,counts;
    auto wrap=[&](auto*ptr,int64_t count){using T=std::remove_pointer_t<decltype(ptr)>;return py::array_t<T>({count},{sizeof(T)},ptr,storage);};
    for(int i=0;i<4;++i) {
        auto&a=view.old[i];auto&s=total.old[i];py::dict d;
        d["width"]=a.width;d["rows"]=wrap(a.rows,s.blocks*a.width);d["prefixes"]=wrap(a.prefixes,s.blocks*a.width);
        d["weights"]=wrap(a.weights,s.blocks*a.width);d["dlog"]=wrap(a.dlog,s.blocks*a.width);
        d["lengths"]=wrap(a.lengths,s.blocks);d["stream_heads"]=wrap(a.heads,s.streams);
        d["resets"]=wrap(a.resets,s.blocks);d["offsets"]=wrap(a.offsets,s.streams+1);
        if(i<3){buckets.append(d);counts.append(s.blocks);}else {
            if(!gradients) {
                d["stream_ids"]=wrap(view.stream_ids,s.streams);d["chunk_ids"]=wrap(view.chunk_ids,s.blocks);
                d["local_offsets"]=wrap(view.local_offsets,s.streams);
                d["short_stream_ids"]=wrap(view.short_stream_ids,total.short_streams);
                d["long_stream_ids"]=wrap(view.long_stream_ids,total.long_streams);
                d["long_chunk_ids"]=wrap(view.long_chunk_ids,total.long_chunks);
                d["long_local_offsets"]=wrap(view.long_local_offsets,total.long_streams);
            }
            result["multi"]=d;
        }
    }
#define VIEW(name,count) split[#name]=wrap(view.name,count)
    VIEW(kr,total.blocks*bk);VIEW(kp,total.blocks*bk);VIEW(kw,total.blocks*bk);VIEW(kd,total.blocks*bk);
    VIEW(kl,total.blocks);VIEW(kreset,total.blocks);VIEW(ktau,total.blocks);
    VIEW(qr,total.tiles*bq);VIEW(qp,total.tiles*bq);VIEW(qw,total.tiles*bq);VIEW(qd,total.tiles*bq);VIEW(qcount,total.tiles*bq);
    VIEW(ql,total.tiles);VIEW(qblock,total.tiles);VIEW(offsets,total.streams+1);VIEW(qoffsets,total.blocks+1);
#undef VIEW
    stats["batch"]=b;stats["heads"]=h;stats["n"]=n;stats["topk"]=topk;stats["tau_rows"]=1;stats["chunk_size"]=width;
    stats["events"]=total.routed+total.legacy;stats["routed_events"]=total.routed;stats["legacy_events"]=total.legacy;
    stats["key_blocks"]=total.blocks;stats["query_tiles"]=total.tiles;stats["state_streams"]=total.streams;
    stats["build_ms"]=milliseconds(begin,built);stats["pack_ms"]=milliseconds(built,done);stats["allocation_ms"]=milliseconds(built,allocated);
    stats["bk"]=bk;stats["bq"]=bq;stats["multi_chunks"]=total.old[3].blocks;stats["multi_streams"]=total.old[3].streams;stats["bucket_blocks"]=counts;
    result["statistics"]=stats;result["buckets"]=buckets;result["split"]=split;result["_storage"]=storage;return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    py::class_<dism_prefill::PlanningWorkers,std::shared_ptr<dism_prefill::PlanningWorkers>>(m,"Planner",py::module_local())
    .def(py::init<int>())
    .def("prepare",&prepare_direct,py::arg("labels"),py::arg("tau"),py::arg("candidates"),
         py::arg("topk"),py::arg("gradients"),py::arg("bk"),py::arg("bq"),py::arg("threshold"),
         py::arg("allocator"),py::arg("hard"),py::arg("documents"));
}
