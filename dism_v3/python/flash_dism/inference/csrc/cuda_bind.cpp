#include "planner.hpp"
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
// Own the control plane and metadata packing natively. Public inputs are already
// projected features; this class does not impose a parameterization or tokenizer.
class NativeCache {
    int heads, r, d, capacity, chunk, position = 0, rebuilds = 0, last_tasks = 0;
    int64_t peak_scratch = 0;
    torch::Tensor keys, values, summaries;
    std::vector<dism_decode::Planner> planners;
    std::vector<int> offsets, mat_counts;
    cudaStream_t owner_stream;
#ifdef DISM_DECODE_TIMING
    std::vector<double> last_timing;
#endif

  public:
    // Explicit CPU rebuild boundary for the GPU-only ordinary-step experiment.
    // The shared key/value allocations were updated in place by the device path.
    void refresh_gpu(torch::Tensor history, int n) {
        c10::cuda::CUDAGuard guard(keys.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream() == owner_stream, "cache stream mismatch");
        check(history, torch::kInt32);
        TORCH_CHECK(history.device() == keys.device() && history.sizes() == torch::IntArrayRef({capacity,3,heads}) && n >= 0 && n <= capacity, "history shape/length");
        auto cpu = history.narrow(0,0,n).cpu();
        const int *data = cpu.data_ptr<int>();
        for (int h=0; h<heads; ++h) {
            std::vector<int> k(n), q(n), reset(n);
            for (int i=0; i<n; ++i) {
                k[i]=data[i*3*heads+h]; q[i]=data[i*3*heads+heads+h];
                reset[i]=data[i*3*heads+2*heads+h];
            }
            auto &p=planners[h];
            p.keys.clear(); p.queries.clear(); p.resets.clear();
            p.initialize(k,q,reset);
        }
        position=n;
        rebuild_all();
    }

    std::vector<torch::Tensor> gpu_snapshot() {
        c10::cuda::CUDAGuard guard(keys.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream() == owner_stream, "cache stream mismatch");
        int nodes=1, edges=1, b=planners[0].interval;
        for (auto &p:planners) {
            nodes=std::max(nodes,int(p.nodes.size()));
            int e=0; for(auto &node:p.nodes) e+=node.next.size();
            edges=std::max(edges,e);
        }
        // Even pitch keeps the final packed (task,next) descriptors 8B aligned.
        nodes=(nodes+1)&~1;
        auto opt=torch::TensorOptions().dtype(torch::kInt32);
        auto topo=torch::full({heads,13,nodes},-1,opt);
        auto transitions=torch::zeros({heads,2,edges},opt);
        auto state=torch::zeros({heads,8},opt);
        auto band=torch::zeros({heads,2,b},opt);
        auto history=torch::zeros({capacity,3,heads},opt);
        auto counts=torch::zeros({summaries.size(0)},opt.dtype(torch::kFloat32));
        auto taus=torch::empty({heads},opt.dtype(torch::kFloat32));
        for(int h=0;h<heads;++h) {
            auto &p=planners[h]; int n=p.nodes.size();
            TORCH_CHECK(p.snapshot_size==position,"export requires rebuilt snapshot");
            int *t=topo.data_ptr<int>()+h*13*nodes;
            int *e=transitions.data_ptr<int>()+h*2*edges;
            int *s=state.data_ptr<int>()+h*8;
            s[0]=position; s[1]=p.state; s[2]=p.match; s[3]=s[4]=-1;
            std::vector<int> inverse(n),end(n);
            for(int i=0;i<n;++i) {inverse[p.order[i]]=i;end[p.order[i]]=i+1;}
            for(int i=n-1;i>0;--i) {int u=p.order[i];end[p.nodes[u].link]=std::max(end[p.nodes[u].link],end[u]);}
            int cursor=0;
            for(int u=0;u<n;++u) {
                auto &node=p.nodes[u];
                t[u]=node.link; t[nodes+u]=node.length; t[2*nodes+u]=node.position;
                t[3*nodes+u]=cursor;
                std::vector<std::pair<int,int>> sorted(node.next.begin(),node.next.end());
                std::sort(sorted.begin(),sorted.end());
                for(auto pair:sorted) {e[cursor]=pair.first;e[edges+cursor]=pair.second;++cursor;}
                t[4*nodes+u]=cursor; t[5*nodes+u]=inverse[u];t[6*nodes+u]=end[u];
                t[7*nodes+u]=p.mat_index[u]<0?-1:offsets[h]+p.mat_index[u];
                t[8*nodes+u]=p.nearest[u];
                t[9*nodes+u]=p.sample_index[u]<0?-1:offsets[h]+mat_counts[h]+p.sample_index[u];
                t[10*nodes+u]=p.order[u];
            }
            std::vector<int> next_valid(n+1,n);
            for(int i=n-1;i>=0;--i) {
                int u=p.order[i],matrix=t[7*nodes+u];
                int id=matrix>=0?capacity+matrix:p.nodes[u].position;
                next_valid[i]=id>=0?i:next_valid[i+1];
                t[11*nodes+2*i]=id;
                t[11*nodes+2*i+1]=next_valid[matrix>=0?end[u]:i+1];
            }
            for(int i=0;i<int(p.band.size());++i) {
                int absolute=position-int(p.band.size())+i;
                band.data_ptr<int>()[(h*2+(position&1))*b+absolute%b]=p.band[i];
            }
            for(int i=0;i<position;++i) {
                history.data_ptr<int>()[i*3*heads+h]=p.keys[i];
                history.data_ptr<int>()[i*3*heads+heads+h]=p.queries[i];
                history.data_ptr<int>()[i*3*heads+2*heads+h]=p.resets[i];
            }
            for(int i=0;i<int(p.materialized.size());++i)
                counts.data_ptr<float>()[offsets[h]+i]=p.subtree_count[p.materialized[i]];
            for(int i=0;i<int(p.samples.size());++i)
                counts.data_ptr<float>()[offsets[h]+mat_counts[h]+i]=p.prefix_weight[p.samples[i]];
            taus.data_ptr<float>()[h]=p.tau;
        }
        return {keys,values,summaries,topo.to(keys.device()),transitions.to(keys.device()),
                state.to(keys.device()),band.to(keys.device()),history.to(keys.device()),
                counts.to(keys.device()),taus.to(keys.device())};
    }
    NativeCache(int h, int r_, int d_, int cap, const std::vector<double> &tau, int b, int sample, int threshold,
                int chunk_, torch::Tensor prototype)
        : heads(h), r(r_), d(d_), capacity(cap), chunk(chunk_), offsets(h, 0), mat_counts(h, 0) {
        TORCH_CHECK(h > 0 && r > 0 && d > 0 && cap > 0 && chunk > 0 && int(tau.size()) == h, "invalid configuration");
        TORCH_CHECK((r == 16 || r == 32) && (d == 32 || d == 64), "supported R16/32 DV32/64");
        TORCH_CHECK(prototype.is_cuda() &&
                        (prototype.scalar_type() == torch::kFloat32 || prototype.scalar_type() == torch::kBFloat16),
                    "prototype must be CUDA FP32/BF16");
        c10::cuda::CUDAGuard guard(prototype.device());
        owner_stream = at::cuda::getCurrentCUDAStream();
        keys = torch::empty({h, cap, r}, prototype.options());
        values = torch::empty({h, cap, d}, prototype.options());
        summaries = torch::empty({0, r * d}, prototype.options().dtype(torch::kFloat32));
        for (auto t : tau)
            planners.emplace_back(r, b, sample, threshold, t);
    }
    void rebuild_all() {
        std::vector<torch::Tensor> outputs;
        int offset = 0;
        for (int h = 0; h < heads; ++h) {
            auto &p = planners[h];
            p.rebuild();
            int n = p.nodes.size();
            int levels = 0;
            if (!p.samples.empty())
                while ((int64_t(1) << levels) < n)
                    ++levels;
            auto host = torch::empty({8 + levels, n}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
            int *data = host.data_ptr<int>();
            for (int i = 0; i < n; ++i) {
                data[i] = p.nodes[i].link;
                data[n + i] = p.nodes[i].length;
                data[2 * n + i] = p.nodes[i].position;
                data[3 * n + i] = p.order[i];
                data[4 * n + i] = p.mat_index[i];
                data[5 * n + i] = p.sample_index[i] < 0 ? -1 : int(p.materialized.size()) + p.sample_index[i];
            }
            for (int i = 0; i < n; ++i) {
                data[6 * n + p.order[i]] = i;
                data[7 * n + p.order[i]] = i + 1;
            }
            for (int i = n - 1; i > 0; --i) {
                int node = p.order[i], parent = p.nodes[node].link;
                data[7 * n + parent] = std::max(data[7 * n + parent], data[7 * n + node]);
            }
            for (int level = 0; level < levels; ++level)
                for (int i = 0; i < n; ++i) {
                    int a = level ? data[(7 + level) * n + i] : p.nodes[i].link;
                    data[(8 + level) * n + i] = level && a >= 0 ? data[(7 + level) * n + a] : a;
                }
            auto topo = host.to(keys.device(), true);
            auto coefficient_host = torch::empty({levels + 1, n}, host.options().dtype(torch::kFloat32));
            float *coefficient_data = coefficient_host.data_ptr<float>();
            for (int i = 0; i < n; ++i)
                coefficient_data[i] = p.edge_weight[i];
            std::vector<double> rho = p.edge_decay, next_rho(n);
            for (int level = 0; level < levels; ++level) {
                for (int i = 0; i < n; ++i)
                    coefficient_data[(level + 1) * n + i] = rho[i];
                for (int i = 0; i < n; ++i) {
                    int a = data[(8 + level) * n + i];
                    next_rho[i] = a < 0 ? 0. : rho[i] * rho[a];
                }
                rho.swap(next_rho);
            }
            auto coefficients = coefficient_host.to(keys.device(), true);
            auto out =
                rebuild(keys[h], values[h], topo, p.materialized.size() + p.samples.size(), p.tau, chunk, coefficients);
            offsets[h] = offset;
            mat_counts[h] = p.materialized.size();
            offset += out.size(0);
            outputs.push_back(out);
            if (out.size(0))
                peak_scratch = std::max(peak_scratch, (int64_t(n) * (8 + int(sizeof(DismPrefix))) +
                                                       ((n + 255) / 256) * int(sizeof(DismPrefix))) *
                                                          std::min(chunk, r * d));
        }
        summaries = torch::cat(outputs);
        ++rebuilds;
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
        for (int h = 0; h < heads; ++h) {
            std::vector<int> k(n), q(n), reset(n);
            for (int i = 0; i < n; ++i) {
                k[i] = data[(i * 3) * heads + h];
                q[i] = data[(i * 3 + 1) * heads + h];
                reset[i] = data[(i * 3 + 2) * heads + h];
            }
            planners[h].initialize(k, q, reset);
        }
        keys.narrow(1, 0, n).copy_(sk);
        values.narrow(1, 0, n).copy_(v);
        position = n;
        rebuild_all();
    }
    torch::Tensor step(torch::Tensor labels, torch::Tensor sk, torch::Tensor sq, torch::Tensor v) {
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
        int64_t nodes = 0;
        for (auto &p : planners)
            nodes += p.nodes.size();
        return {{"position", position},
                {"capacity", capacity},
                {"raw_bytes", keys.nbytes() + values.nbytes()},
                {"summary_bytes", summaries.nbytes()},
                {"summary_matrices", summaries.size(0)},
                {"rebuilds", rebuilds},
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
    bind_gpu_planner(m);
    m.def("rebuild", [](torch::Tensor k, torch::Tensor v, torch::Tensor topology, int matrices, float tau, int chunk) {
        return rebuild(k, v, topology, matrices, tau, chunk);
    });
    m.def("query", &query);
    pybind11::class_<NativeCache>(m, "NativeCache")
        .def(pybind11::init<int, int, int, int, const std::vector<double> &, int, int, int, int, torch::Tensor>())
        .def("step", &NativeCache::step, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("prime", &NativeCache::prime, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("gpu_snapshot", &NativeCache::gpu_snapshot)
        .def("refresh_gpu", &NativeCache::refresh_gpu)
#ifdef DISM_DECODE_TIMING
        .def("timings", &NativeCache::timings)
#endif
        .def("stats", &NativeCache::stats);
    pybind11::class_<LinearCache>(m, "LinearCache")
        .def(pybind11::init<int, int, int, int, const std::vector<double> &, torch::Tensor>())
        .def("step", &LinearCache::step, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("prime", &LinearCache::prime, pybind11::call_guard<pybind11::gil_scoped_release>());
}
