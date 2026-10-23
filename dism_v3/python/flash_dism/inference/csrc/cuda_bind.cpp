#include "probe/summary_program_host.hpp"
#include "planner.hpp"
#include "decode_rebuild.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <optional>
#include "prefix_type.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <torch/extension.h>
#ifdef DISM_DECODE_TIMING
#include <chrono>
#endif

extern "C" void launch_rebuild(const void *, const void *, const int *, float *, float *, int, int, int, int, int, int,
                               float, bool, cudaStream_t);
extern "C" void launch_query(void *, void *, const void *, const void *, const void *, const float *, const int *,
                             const float *, const float *, float *, int, int, int, int, int, int, bool, bool,
                             cudaStream_t);
extern "C" void launch_parallel_rebuild(const void *, const void *, const int *, const float *, DismPrefix *,
                                        DismPrefix *, float *, float *, float *, int, int, int, int, int, int, float,
                                        bool, cudaStream_t);
extern "C" void launch_linear_prime(const int *, int *, int *, int, int, int, cudaStream_t);
extern "C" void launch_linear(void *, void *, int *, const int *, int *, const int *, const void *, const void *,
                              const void *, const float *, float *, int, int, int, int, int, bool, cudaStream_t);

void check(torch::Tensor x, torch::ScalarType type) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == type, "invalid device/dtype/layout");
}
torch::Tensor rebuild(torch::Tensor k, torch::Tensor v, torch::Tensor topology, int matrices, float tau, int chunk,
                      torch::Tensor coefficients = {}) {
    TORCH_CHECK(k.scalar_type() == torch::kFloat32 || k.scalar_type() == torch::kBFloat16, "cache dtype");
    check(k, k.scalar_type());
    check(v, k.scalar_type());
    check(topology, torch::kInt32);
    TORCH_CHECK(k.dim() == 2 && v.dim() == 2 && k.size(0) == v.size(0), "cache dimensions");
    TORCH_CHECK(topology.dim() == 2 && (topology.size(0) == 6 || topology.size(0) >= 8) && chunk > 0 && matrices >= 0,
                "topology dimensions");
    TORCH_CHECK(k.device() == v.device() && k.device() == topology.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(k.device());
    int n = topology.size(1), r = k.size(1), d = v.size(1), elements = r * d + (topology.size(0) == 6 ? 1 : 0);
    auto out = torch::empty({matrices, elements}, k.options().dtype(torch::kFloat32));
    if (matrices == 0)
        return out;
    if (topology.size(0) >= 8) {
        int width = std::min(chunk, elements), levels = topology.size(0) - 8;
        check(coefficients, torch::kFloat32);
        TORCH_CHECK(coefficients.device() == k.device() && coefficients.numel() == (levels + 1) * n,
                    "rebuild coefficient shape/device");
        auto prefix =
            torch::empty({width, n}, out.options().dtype(sizeof(DismPrefix) == 8 ? torch::kFloat64 : torch::kFloat32));
        auto totals = torch::empty({width, (n + 255) / 256}, prefix.options());
        auto ping = torch::empty({n, width}, out.options()), pong = torch::empty_like(ping);
        for (int start = 0; start < elements; start += chunk)
            launch_parallel_rebuild(k.data_ptr(), v.data_ptr(), topology.data_ptr<int>(),
                                    coefficients.data_ptr<float>(), prefix.data_ptr<DismPrefix>(),
                                    totals.data_ptr<DismPrefix>(), ping.data_ptr<float>(), pong.data_ptr<float>(),
                                    out.data_ptr<float>(), n, r, d, start, std::min(chunk, elements - start), levels,
                                    tau, k.scalar_type() == torch::kBFloat16, at::cuda::getCurrentCUDAStream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    auto scratch = torch::empty({n, std::min(chunk, elements)}, out.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int start = 0; start < elements; start += chunk)
        launch_rebuild(k.data_ptr(), v.data_ptr(), topology.data_ptr<int>(), scratch.data_ptr<float>(),
                       out.data_ptr<float>(), n, r, d, start, std::min(chunk, elements - start), scratch.size(1), tau,
                       k.scalar_type() == torch::kBFloat16, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor query(torch::Tensor k, torch::Tensor v, torch::Tensor nk, torch::Tensor nv, torch::Tensor q,
                    torch::Tensor summaries, torch::Tensor ids, torch::Tensor coeff, torch::Tensor fallback,
                    int position) {
    for (auto x : {k, v, nk, nv, q, summaries, coeff, fallback}) {
        check(x, torch::kFloat32);
        TORCH_CHECK(x.device() == k.device(), "device mismatch");
    }
    check(ids, torch::kInt32);
    TORCH_CHECK(ids.device() == k.device(), "device mismatch");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "cache rank");
    int h = k.size(0), capacity = k.size(1), r = k.size(2), d = v.size(2);
    TORCH_CHECK((r == 16 || r == 32) && (d == 32 || d == 64), "supported R16/32 DV32/64");
    TORCH_CHECK(v.size(0) == h && v.size(1) == capacity && position >= 0 && position < capacity,
                "cache shape/position");
    TORCH_CHECK(nk.sizes() == q.sizes() && q.dim() == 2 && q.size(0) == h && q.size(1) == r, "query shape");
    TORCH_CHECK(nv.dim() == 2 && nv.size(0) == h && nv.size(1) == d, "value shape");
    TORCH_CHECK(ids.dim() == 3 && ids.size(0) == h && ids.size(2) == 2, "task shape");
    TORCH_CHECK(coeff.dim() == 2 && coeff.size(0) == h && coeff.size(1) == ids.size(1), "coefficient shape");
    TORCH_CHECK(fallback.numel() == h && summaries.dim() == 2 && summaries.size(1) == r * d + 1, "summary shape");
    c10::cuda::CUDAGuard guard(k.device());
    auto out = torch::empty({h, d}, k.options());
    launch_query(k.data_ptr<float>(), v.data_ptr<float>(), nk.data_ptr<float>(), nv.data_ptr<float>(),
                 q.data_ptr<float>(), summaries.data_ptr<float>(), ids.data_ptr<int>(), coeff.data_ptr<float>(),
                 fallback.data_ptr<float>(), out.data_ptr<float>(), h, capacity, position, r, d, ids.size(1), false,
                 false, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
// Reusable pinned CPU storage. Reusing an allocation directly bypasses the
// caching allocator's reuse checks, so wait for its prior H2D before writing.
struct PinnedArena {
    torch::Tensor storage;
    at::cuda::CUDAEvent copied;
    bool pending=false;
    torch::Tensor acquire(int64_t words) {
        if(pending) { copied.synchronize(); pending=false; }
        if(!storage.defined() || storage.numel()<words) {
            int64_t grown=storage.defined()?storage.numel()+storage.numel()/2:0;
            int64_t capacity_words=(std::max(words,grown)+4095)/4096*4096;
            storage=torch::empty({capacity_words},torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        }
        return storage.narrow(0,0,words);
    }
    torch::Tensor upload(const torch::Tensor &host,const torch::Device &device) {
        auto result=host.to(device,true);
        copied.record(at::cuda::getCurrentCUDAStream());pending=true;
        return result;
    }
};
// Own the control plane and metadata packing natively. Public inputs are already
// projected features; this class does not impose a parameterization or tokenizer.
class NativeCache {
    int heads, r, d, capacity, chunk, position = 0, rebuilds = 0, last_tasks = 0;
    int64_t peak_scratch = 0;
    torch::Tensor keys, values, summaries;
    std::shared_ptr<std::vector<dism_decode::Planner>> planner_store=std::make_shared<std::vector<dism_decode::Planner>>();
    std::vector<dism_decode::Planner> &planners=*planner_store;
    std::vector<int> offsets, mat_counts;
    cudaStream_t owner_stream;
    int worker_count=1, history_uploaded=0, last_refresh_tokens=0;
    bool last_refresh_band_reused=false,summary_pending=false;
    int pending_matrices=0;
    std::shared_ptr<dism_prefill::PlanningWorkers> workers;
    std::vector<dism_decode::RebuildWorkspace> workspaces;
    PinnedArena summary_host, snapshot_host, refresh_host;
    torch::Tensor gpu_history;
#ifdef DISM_DECODE_TIMING
    std::vector<double> last_timing;
#endif

  public:
    // Explicit CPU rebuild boundary for the GPU-only ordinary-step experiment.
    // The shared key/value allocations were updated in place by the device path.
    void refresh_gpu(torch::Tensor history, int n, std::optional<torch::Tensor> device_band=std::nullopt,bool build_vectors=true) {
        c10::cuda::CUDAGuard guard(keys.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream()==owner_stream,"cache stream mismatch");
        check(history,torch::kInt32);
        TORCH_CHECK(history.device()==keys.device() && history.sizes()==torch::IntArrayRef({capacity,3,heads})
                    && n>=0 && n<=capacity,"history shape/length");
        const int interval=planners[0].interval;
        if(device_band) {
            check(*device_band,torch::kInt32);
            TORCH_CHECK(device_band->device()==keys.device() &&
                device_band->sizes()==torch::IntArrayRef({heads,2,interval}),"band shape/device");
        }
        // The optimized GPU-owned path is append-only. The old two-argument
        // API still replaces complete history and recomputes its band.
        bool incremental=device_band && gpu_history.defined() &&
                         gpu_history.data_ptr()==history.data_ptr() && n>=position;
        int start=incremental?position:0, added=n-start;
        int64_t label_words=int64_t(added)*3*heads;
        auto host=refresh_host.acquire(label_words+(device_band?int64_t(heads)*interval:0));
        auto ints=host.view(torch::kInt32);
        if(added) ints.narrow(0,0,label_words).view({added,3,heads}).copy_(history.narrow(0,start,added),true);
        if(device_band) ints.narrow(0,label_words,int64_t(heads)*interval).view({heads,interval})
            .copy_(device_band->select(1,n&1),true);
        C10_CUDA_CHECK(cudaStreamSynchronize(owner_stream));
        const int *data=ints.data_ptr<int>();
        workers->run(heads,[&](size_t h) {
            auto &p=planners[h];
            if(!incremental) {p.keys.clear();p.queries.clear();p.resets.clear();p.reset_sam();}
            if(int(p.keys.size())!=start) throw std::logic_error("CPU/GPU history positions disagree");
            p.keys.resize(n);p.queries.resize(n);p.resets.resize(n);
            for(int i=0;i<added;++i) {
                int reset=data[int64_t(i)*3*heads+2*heads+h];
                if(reset!=0 && reset!=1) throw std::invalid_argument("reset must be 0/1");
                p.keys[start+i]=data[int64_t(i)*3*heads+h];
                p.queries[start+i]=data[int64_t(i)*3*heads+heads+h];
                p.resets[start+i]=reset;
            }
            if(device_band) {
                int begin=std::max(0,n-interval);p.band.resize(n-begin);
                const int *ring=data+label_words+h*interval;
                for(int pos=begin;pos<n;++pos) {
                    int length=ring[pos%interval];
                    if(length<0 || length>pos+1) throw std::invalid_argument("invalid GPU band length");
                    p.band[pos-begin]=length;
                }
            } else p.restore_band();
        });
        position=n;gpu_history=history;history_uploaded=n;
        last_refresh_tokens=added;last_refresh_band_reused=device_band.has_value();
        rebuild_all(build_vectors);
    }

    std::shared_ptr<SummaryProgram> summary_program(int block,int tile,int slots) {
        TORCH_CHECK(at::cuda::getCurrentCUDAStream()==owner_stream,"cache stream mismatch");
        TORCH_CHECK(position>0,"nonempty planner snapshot required");
        std::vector<double> tau;for(const auto &p:planners)tau.push_back(p.tau);
        return std::make_shared<SummaryProgram>(torch::Tensor(),tau,r,d,std::min(block,2*position),tile,slots,true,true,false,planner_store,&workspaces);
    }
    void install_summary(torch::Tensor output) {
        TORCH_CHECK(at::cuda::getCurrentCUDAStream()==owner_stream,"cache stream mismatch");
        check(output,torch::kFloat32);
        TORCH_CHECK(output.device()==keys.device() && output.sizes()==torch::IntArrayRef({summary_pending?pending_matrices:summaries.size(0),r*d}),"summary output shape/device");
        summaries=output;summary_pending=false;
    }

    std::vector<torch::Tensor> gpu_snapshot() {
        TORCH_CHECK(!summary_pending,"persistent summary output must be installed before export");
        c10::cuda::CUDAGuard guard(keys.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream()==owner_stream,"cache stream mismatch");
        int nodes=1,edges=1,interval=planners[0].interval;
        for(int h=0;h<heads;++h) {
            TORCH_CHECK(planners[h].snapshot_size==position,"export requires rebuilt snapshot");
            nodes=std::max(nodes,int(planners[h].nodes.size()));edges=std::max(edges,workspaces[h].edges);
        }
        nodes=(nodes+1)&~1; // int2 descriptors require even pitch.
        if(!gpu_history.defined()) {
            gpu_history=torch::zeros({capacity,3,heads},keys.options().dtype(torch::kInt32));
            history_uploaded=0;
        }
        int tail=position-history_uploaded;
        TORCH_CHECK(tail>=0,"GPU history ahead of CPU snapshot");
        int64_t topo_words=int64_t(heads)*13*nodes,edge_words=int64_t(heads)*2*edges;
        int64_t state_at=topo_words+edge_words,band_at=state_at+heads*8;
        int64_t counts_at=band_at+int64_t(heads)*2*interval,taus_at=counts_at+summaries.size(0);
        int64_t history_at=taus_at+heads,words=history_at+int64_t(tail)*3*heads;
        auto host=snapshot_host.acquire(words);
        auto raw=host.data_ptr<float>();auto ints=reinterpret_cast<int*>(raw);
        workers->run(heads,[&](size_t h) {
            const auto &p=planners[h];
            workspaces[h].pack_snapshot(p,nodes,edges,capacity,offsets[h],mat_counts[h],
                ints+h*13*nodes,ints+topo_words+h*2*edges,ints+state_at+h*8,
                ints+band_at+h*2*interval,raw+counts_at,raw+taus_at+h);
            for(int i=0;i<tail;++i) {
                int j=history_uploaded+i;
                ints[history_at+int64_t(i)*3*heads+h]=p.keys[j];
                ints[history_at+int64_t(i)*3*heads+heads+h]=p.queries[j];
                ints[history_at+int64_t(i)*3*heads+2*heads+h]=p.resets[j];
            }
        },true);
        auto packed=snapshot_host.upload(host,keys.device());
        auto packed_i=packed.view(torch::kInt32);
        if(tail) gpu_history.narrow(0,history_uploaded,tail).copy_(
            packed_i.narrow(0,history_at,int64_t(tail)*3*heads).view({tail,3,heads}));
        history_uploaded=position;
        return {keys,values,summaries,
            packed_i.narrow(0,0,topo_words).view({heads,13,nodes}),
            packed_i.narrow(0,topo_words,edge_words).view({heads,2,edges}),
            packed_i.narrow(0,state_at,heads*8).view({heads,8}),
            packed_i.narrow(0,band_at,int64_t(heads)*2*interval).view({heads,2,interval}),
            gpu_history,packed.narrow(0,counts_at,summaries.size(0)),packed.narrow(0,taus_at,heads)};
    }
    NativeCache(int h, int r_, int d_, int cap, const std::vector<double> &tau, int b, int sample, int threshold,
                int chunk_, torch::Tensor prototype, int rebuild_workers=8)
        : heads(h), r(r_), d(d_), capacity(cap), chunk(chunk_), offsets(h, 0), mat_counts(h, 0) {
        TORCH_CHECK(h > 0 && r > 0 && d > 0 && cap > 0 && chunk > 0 && int(tau.size()) == h, "invalid configuration");
        TORCH_CHECK((r == 16 || r == 32) && (d == 32 || d == 64), "supported R16/32 DV32/64");
        TORCH_CHECK(prototype.is_cuda() &&
                        (prototype.scalar_type() == torch::kFloat32 || prototype.scalar_type() == torch::kBFloat16),
                    "prototype must be CUDA FP32/BF16");
        c10::cuda::CUDAGuard guard(prototype.device());
        TORCH_CHECK(rebuild_workers>=1 && rebuild_workers<=64,"rebuild_workers must be in [1,64]");
        worker_count=std::min(h,rebuild_workers);
        workers=std::make_shared<dism_prefill::PlanningWorkers>(worker_count);
        workspaces.resize(h);
        owner_stream = at::cuda::getCurrentCUDAStream();
        keys = torch::empty({h, cap, r}, prototype.options());
        values = torch::empty({h, cap, d}, prototype.options());
        summaries = torch::empty({0, r * d}, prototype.options().dtype(torch::kFloat32));
        for (auto t : tau)
            planners.emplace_back(r, b, sample, threshold, t);
        workers->run(heads,[&](size_t h){workspaces[h].prepare(planners[h]);});
    }
    void rebuild_all(bool build_vectors=true) {
        workers->run(heads,[&](size_t h) {planners[h].rebuild();workspaces[h].prepare(planners[h]);});
        std::vector<int64_t> starts(heads),coefficient_starts(heads);
        int64_t words=0;int matrices=0;
        for(int h=0;h<heads;++h) {
            const auto &p=planners[h];int n=p.nodes.size(),levels=workspaces[h].levels;
            offsets[h]=matrices;mat_counts[h]=p.materialized.size();
            int count=p.materialized.size()+p.samples.size();matrices+=count;
            starts[h]=words;
            if(count) words+=int64_t(8+levels)*n;
            coefficient_starts[h]=words;
            if(count) words+=int64_t(levels+1)*n;
        }
        if(!build_vectors) {
            // Caller installs a persistent-program output before exporting metadata.
            pending_matrices=matrices;
            summary_pending=true;++rebuilds;return;
        }
        // No GPU topology/coefficients or transfer for empty-summary heads.
        if(!matrices) {
            summaries=torch::empty({0,r*d},keys.options().dtype(torch::kFloat32));summary_pending=false;++rebuilds;return;
        }
        auto host=summary_host.acquire(words);auto raw=host.data_ptr<float>();
        workers->run(heads,[&](size_t h) {
            const auto &p=planners[h];
            if(p.materialized.empty() && p.samples.empty()) return;
            workspaces[h].pack_summary(p,reinterpret_cast<int*>(raw+starts[h]),raw+coefficient_starts[h]);
        },true);
        auto packed=summary_host.upload(host,keys.device());
        std::vector<torch::Tensor> outputs;
        // Only this thread touches CUDA: preserve head order, stream and math.
        for(int h=0;h<heads;++h) {
            const auto &p=planners[h];int n=p.nodes.size(),levels=workspaces[h].levels;
            int count=p.materialized.size()+p.samples.size();
            if(!count) {outputs.push_back(torch::empty({0,r*d},summaries.options()));continue;}
            auto topo=packed.narrow(0,starts[h],int64_t(8+levels)*n).view(torch::kInt32).view({8+levels,n});
            auto coefficients=packed.narrow(0,coefficient_starts[h],int64_t(levels+1)*n).view({levels+1,n});
            outputs.push_back(rebuild(keys[h],values[h],topo,count,p.tau,chunk,coefficients));
            peak_scratch=std::max(peak_scratch,(int64_t(n)*(8+int(sizeof(DismPrefix)))+
                ((n+255)/256)*int(sizeof(DismPrefix)))*std::min(chunk,r*d));
        }
        summaries=torch::cat(outputs);summary_pending=false;++rebuilds;
    }
    void prime(torch::Tensor labels, torch::Tensor sk, torch::Tensor v) {
        c10::cuda::CUDAGuard guard(keys.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream() == owner_stream, "cache stream mismatch");
        TORCH_CHECK(position == 0, "prime requires empty cache");
        for (auto x : {sk, v}) {
            check(x, keys.scalar_type());
            TORCH_CHECK(x.device() == keys.device(), "device mismatch");
        }
        TORCH_CHECK(sk.dim() == 3 && sk.size(0) == heads && sk.size(2) == r, "prime sk must be [BH,N,R]");
        int n = sk.size(1);
        TORCH_CHECK(n > 0 && n <= capacity && v.dim() == 3 && v.size(0) == heads && v.size(1) == n && v.size(2) == d,
                    "prime dimensions");
        TORCH_CHECK(labels.scalar_type() == torch::kInt32 && labels.is_contiguous() && labels.dim() == 3 &&
                        labels.size(0) == n && labels.size(1) == 3 && labels.size(2) == heads,
                    "prime labels must be [N,3,BH]");
        TORCH_CHECK(!labels.is_cuda() || labels.device() == keys.device(), "label device mismatch");
        auto cpu = labels.cpu();
        auto data = cpu.data_ptr<int>();
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < heads; ++h)
                TORCH_CHECK(data[(i * 3 + 2) * heads + h] == 0 || data[(i * 3 + 2) * heads + h] == 1, "invalid reset");
        workers->run(heads,[&](size_t h) {
            std::vector<int> k(n), q(n), reset(n);
            for (int i = 0; i < n; ++i) {
                k[i] = data[(i * 3) * heads + h];
                q[i] = data[(i * 3 + 1) * heads + h];
                reset[i] = data[(i * 3 + 2) * heads + h];
            }
            planners[h].initialize(k, q, reset);
        });
        keys.narrow(1, 0, n).copy_(sk);
        values.narrow(1, 0, n).copy_(v);
        position = n;
        if(gpu_history.defined()) {gpu_history.narrow(0,0,n).copy_(labels);history_uploaded=n;}
        rebuild_all();
    }
    torch::Tensor step(torch::Tensor labels, torch::Tensor sk, torch::Tensor sq, torch::Tensor v) {
        TORCH_CHECK(!summary_pending,"persistent summary output must be installed before decode");
#ifdef DISM_DECODE_TIMING
        using Clock = std::chrono::steady_clock;
        auto previous = Clock::now();
        last_timing.clear();
        auto stamp = [&]() {
            auto now = Clock::now();
            last_timing.push_back(std::chrono::duration<double, std::micro>(now-previous).count());
            previous = now;
        };
#endif
        c10::cuda::CUDAGuard guard(keys.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream() == owner_stream, "cache is bound to its creating CUDA stream");
        TORCH_CHECK(position < capacity, "cache capacity exceeded");
        for (auto x : {sk, sq, v}) {
            check(x, keys.scalar_type());
            TORCH_CHECK(x.device() == keys.device(), "device mismatch");
        }
        TORCH_CHECK(sk.dim() == 2 && sk.size(0) == heads && sk.size(1) == r && sq.sizes() == sk.sizes(),
                    "soft feature shape");
        TORCH_CHECK(v.dim() == 2 && v.size(0) == heads && v.size(1) == d, "value shape");
        TORCH_CHECK(labels.scalar_type() == torch::kInt32 && labels.is_contiguous() && labels.dim() == 2 &&
                        labels.size(0) == 3 && labels.size(1) == heads,
                    "labels must be int32 [3,BH]: key,query,reset");
        TORCH_CHECK(!labels.is_cuda() || labels.device() == keys.device(), "label device mismatch");
        // Single bulk D2H and synchronization when labels originate on device.
#ifdef DISM_DECODE_TIMING
        stamp(); // validation
#endif
        auto cpu = labels.cpu();
#ifdef DISM_DECODE_TIMING
        stamp(); // D2H allocation, enqueue, and CPU-visible completion
#endif
        const int *input = cpu.data_ptr<int>();
        for (int h = 0; h < heads; ++h)
            TORCH_CHECK(input[2 * heads + h] == 0 || input[2 * heads + h] == 1, "reset must be 0/1");
        if (planners[0].needs_rebuild())
            rebuild_all();
#ifdef DISM_DECODE_TIMING
        stamp(); // rebuild (if due), validation
#endif
        std::vector<std::vector<dism_decode::Task>> plans;
        int tasks = 1;
        for (int h = 0; h < heads; ++h) {
            plans.push_back(planners[h].append(input[h], input[heads + h], input[2 * heads + h]));
            tasks = std::max(tasks, int(plans.back().size()));
        }
        // One pinned allocation and one H2D: [int32 ids][float weights][inverse denominator].
#ifdef DISM_DECODE_TIMING
        stamp(); // CPU planner
#endif
        int64_t count = int64_t(heads) * tasks;
        auto host =
            torch::empty({3 * count + heads}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        std::memset(host.data_ptr(), 0, host.nbytes());
        int *ids = reinterpret_cast<int *>(host.data_ptr());
        float *weights = host.data_ptr<float>() + 2 * count;
        float *fallback = weights + count;
        for (int h = 0; h < heads; ++h) {
            fallback[h] = planners[h].inv_denominator;
            for (int t = 0; t < int(plans[h].size()); ++t) {
                auto task = plans[h][t];
                int64_t slot = int64_t(h) * tasks + t;
                int index = task.index;
                if (task.kind)
                    index += offsets[h] + (task.kind == 2 ? mat_counts[h] : 0);
                ids[2 * slot] = task.kind;
                ids[2 * slot + 1] = index;
                weights[slot] = task.coefficient;
            }
        }
        // Tensor copy records the pinned allocation with the caching allocator.
#ifdef DISM_DECODE_TIMING
        stamp(); // pinned allocation and metadata packing
#endif
        auto metadata = host.to(keys.device(), true);
#ifdef DISM_DECODE_TIMING
        stamp(); // H2D allocation and enqueue, NOT GPU completion
#endif
        auto out = torch::empty({heads, d}, summaries.options());
        auto data = metadata.data_ptr<float>();
        launch_query(keys.data_ptr(), values.data_ptr(), sk.data_ptr(), v.data_ptr(), sq.data_ptr(),
                     summaries.data_ptr<float>(), reinterpret_cast<int *>(data), data + 2 * count, data + 3 * count,
                     out.data_ptr<float>(), heads, capacity, position, r, d, tasks,
                     keys.scalar_type() == torch::kBFloat16, true, owner_stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        ++position;
        last_tasks = tasks;
#ifdef DISM_DECODE_TIMING
        stamp(); // output allocation, launch, cleanup bookkeeping
#endif
        return out;
    }
#ifdef DISM_DECODE_TIMING
    std::vector<double> timings() const { return last_timing; }
#endif
    std::map<std::string, int64_t> stats() const {
        int64_t nodes = 0,extended=0,pinned=0;
        for (auto &p : planners) {nodes += p.nodes.size();extended+=p.last_extended_keys;}
        for (const auto *arena:{&summary_host,&snapshot_host,&refresh_host})
            if(arena->storage.defined()) pinned+=arena->storage.nbytes();
        return {{"position", position},
                {"capacity", capacity},
                {"raw_bytes", keys.nbytes() + values.nbytes()},
                {"summary_bytes", summaries.nbytes()},
                {"summary_matrices", summaries.size(0)},
                {"rebuilds", rebuilds},
                {"rebuild_workers", worker_count},
                {"last_sam_extended_keys", extended},
                {"last_refresh_tokens", last_refresh_tokens},
                {"last_refresh_band_reused", last_refresh_band_reused},
                {"pinned_rebuild_capacity_bytes", pinned},
                {"last_tasks_per_head", last_tasks},
                {"sam_nodes", nodes},
                {"max_single_rebuild_scratch_bytes", peak_scratch}};
    }
};
class LinearCache {
    int heads, r, d, capacity, position = 0;
    torch::Tensor keys, values, labels, previous, next, tau;
    cudaStream_t stream;

  public:
    LinearCache(int h, int r_, int d_, int cap, const std::vector<double> &taus, torch::Tensor prototype)
        : heads(h), r(r_), d(d_), capacity(cap) {
        TORCH_CHECK(h > 0 && cap > 0 && (r == 16 || r == 32) && (d == 32 || d == 64) && int(taus.size()) == h,
                    "invalid dimensions");
        TORCH_CHECK(prototype.is_cuda() &&
                        (prototype.scalar_type() == torch::kFloat32 || prototype.scalar_type() == torch::kBFloat16),
                    "invalid prototype");
        for (auto t : taus)
            TORCH_CHECK(std::isfinite(t) && t >= 0 && t <= std::numeric_limits<float>::max(), "invalid tau");
        c10::cuda::CUDAGuard guard(prototype.device());
        stream = at::cuda::getCurrentCUDAStream();
        keys = torch::empty({h, cap, r}, prototype.options());
        values = torch::empty({h, cap, d}, prototype.options());
        labels = torch::empty({h, cap}, prototype.options().dtype(torch::kInt32));
        previous = torch::empty_like(labels);
        next = torch::empty_like(labels);
        tau = torch::tensor(taus, torch::TensorOptions().dtype(torch::kFloat32)).to(prototype.device());
    }
    void validate(torch::Tensor packed, torch::Tensor sk, torch::Tensor v, bool prime) {
        TORCH_CHECK(at::cuda::getCurrentCUDAStream() == stream, "cache stream mismatch");
        for (auto x : {sk, v}) {
            check(x, keys.scalar_type());
            TORCH_CHECK(x.device() == keys.device(), "device mismatch");
        }
        check(packed, torch::kInt32);
        TORCH_CHECK(packed.device() == keys.device(), "device mismatch");
        if (prime) {
            TORCH_CHECK(position == 0 && sk.dim() == 3 && sk.size(0) == heads && sk.size(2) == r, "prime shape/cache");
            int n = sk.size(1);
            TORCH_CHECK(n > 0 && n <= capacity && v.sizes() == torch::IntArrayRef({heads, n, d}), "prime value shape");
            TORCH_CHECK(packed.sizes() == torch::IntArrayRef({n, 3, heads}), "prime labels shape");
        } else {
            TORCH_CHECK(position < capacity && sk.sizes() == torch::IntArrayRef({heads, r}) &&
                            v.sizes() == torch::IntArrayRef({heads, d}),
                        "step shape/capacity");
            TORCH_CHECK(packed.sizes() == torch::IntArrayRef({3, heads}), "step labels shape");
        }
    }
    void prime(torch::Tensor packed, torch::Tensor sk, torch::Tensor v) {
        c10::cuda::CUDAGuard guard(keys.device());
        validate(packed, sk, v, true);
        int n = sk.size(1);
        keys.narrow(1, 0, n).copy_(sk);
        values.narrow(1, 0, n).copy_(v);
        launch_linear_prime(packed.data_ptr<int>(), labels.data_ptr<int>(), previous.data_ptr<int>(), heads, n,
                            capacity, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        position = n;
    }
    torch::Tensor step(torch::Tensor packed, torch::Tensor sk, torch::Tensor sq, torch::Tensor v) {
        c10::cuda::CUDAGuard guard(keys.device());
        validate(packed, sk, v, false);
        check(sq, keys.scalar_type());
        TORCH_CHECK(sq.device() == keys.device() && sq.sizes() == sk.sizes(), "query shape/device");
        auto out = torch::empty({heads, d}, keys.options().dtype(torch::kFloat32));
        launch_linear(keys.data_ptr(), values.data_ptr(), labels.data_ptr<int>(), previous.data_ptr<int>(),
                      next.data_ptr<int>(), packed.data_ptr<int>(), sk.data_ptr(), v.data_ptr(), sq.data_ptr(),
                      tau.data_ptr<float>(), out.data_ptr<float>(), heads, capacity, position, r, d,
                      keys.scalar_type() == torch::kBFloat16, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        std::swap(previous, next);
        ++position;
        return out;
    }
};
void bind_gpu_planner(pybind11::module_ &m);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    bind_summary_program(m);
    bind_gpu_planner(m);
    m.def("rebuild", [](torch::Tensor k, torch::Tensor v, torch::Tensor topology, int matrices, float tau, int chunk) {
        return rebuild(k, v, topology, matrices, tau, chunk);
    });
    m.def("query", &query);
    pybind11::class_<NativeCache>(m, "NativeCache")
        .def(pybind11::init<int, int, int, int, const std::vector<double> &, int, int, int, int, torch::Tensor, int>(),
             pybind11::arg("heads"),pybind11::arg("r"),pybind11::arg("d"),pybind11::arg("capacity"),
             pybind11::arg("tau"),pybind11::arg("interval"),pybind11::arg("sample"),pybind11::arg("threshold"),
             pybind11::arg("chunk"),pybind11::arg("prototype"),pybind11::arg("rebuild_workers")=8)
        .def("step", &NativeCache::step, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("prime", &NativeCache::prime, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("summary_program", &NativeCache::summary_program,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("install_summary", &NativeCache::install_summary,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("gpu_snapshot", &NativeCache::gpu_snapshot,pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("refresh_gpu", &NativeCache::refresh_gpu,pybind11::arg("history"),pybind11::arg("position"),
             pybind11::arg("band")=pybind11::none(),pybind11::arg("build_vectors")=true,pybind11::call_guard<pybind11::gil_scoped_release>())
#ifdef DISM_DECODE_TIMING
        .def("timings", &NativeCache::timings)
#endif
        .def("stats", &NativeCache::stats);
    pybind11::class_<LinearCache>(m, "LinearCache")
        .def(pybind11::init<int, int, int, int, const std::vector<double> &, torch::Tensor>())
        .def("step", &LinearCache::step, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("prime", &LinearCache::prime, pybind11::call_guard<pybind11::gil_scoped_release>());
}
