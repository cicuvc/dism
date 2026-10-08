#include "gpu_planner_params.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

extern "C" void launch_gpu_planner(GpuPlannerArgs, const int *, const void *, const void *, const void *,
                                   bool, cudaStream_t);

class GpuPlanner {
    std::vector<torch::Tensor> snapshot;
    torch::Tensor marks, slots, tasks, coefficients, output;
    GpuPlannerArgs args{};
    cudaStream_t stream;

  public:
    GpuPlanner(std::vector<torch::Tensor> tensors, int position) : snapshot(std::move(tensors)) {
        TORCH_CHECK(snapshot.size() == 10, "snapshot fields");
        auto &k = snapshot[0];
        c10::cuda::CUDAGuard guard(k.device());
        stream = at::cuda::getCurrentCUDAStream();
        args.heads = k.size(0);
        args.capacity = k.size(1);
        args.r = k.size(2);
        args.d = snapshot[1].size(2);
        args.interval = snapshot[6].size(2);
        args.snapshot = position;
        args.nodes = snapshot[3].size(2);
        args.edge_capacity = snapshot[4].size(2);
        // Each raw position and each summary can appear at most once after
        // epoch-based merging. This bound cannot overflow for a valid snapshot.
        args.task_capacity = args.capacity + snapshot[2].size(0);
        auto io = k.options().dtype(torch::kInt32);
        marks = torch::full({args.heads, args.task_capacity}, -1, io);
        slots = torch::empty_like(marks);
        tasks = torch::empty_like(marks);
        coefficients = torch::empty({args.heads, args.task_capacity}, k.options().dtype(torch::kFloat32));
        output = torch::empty({args.heads, args.d}, coefficients.options());
        args.keys = k.data_ptr();
        args.values = snapshot[1].data_ptr();
        args.summaries = snapshot[2].data_ptr<float>();
        args.topology = snapshot[3].data_ptr<int>();
        args.edges = snapshot[4].data_ptr<int>();
        args.state = snapshot[5].data_ptr<int>();
        args.band = snapshot[6].data_ptr<int>();
        args.history = snapshot[7].data_ptr<int>();
        args.counts = snapshot[8].data_ptr<float>();
        args.tau = snapshot[9].data_ptr<float>();
        args.marks = marks.data_ptr<int>();
        args.slots = slots.data_ptr<int>();
        args.tasks = tasks.data_ptr<int>();
        args.coefficients = coefficients.data_ptr<float>();
        args.output = output.data_ptr<float>();
    }
    torch::Tensor step(torch::Tensor labels, torch::Tensor sk, torch::Tensor sq, torch::Tensor v) {
        auto &k = snapshot[0];
        c10::cuda::CUDAGuard guard(k.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream() == stream, "cache stream mismatch");
        TORCH_CHECK(labels.device() == k.device() && labels.scalar_type() == torch::kInt32 &&
                        labels.is_contiguous() && labels.sizes() == torch::IntArrayRef({3, args.heads}),
                    "labels must be CUDA int32 [3,H]");
        for (auto x : {sk, sq, v})
            TORCH_CHECK(x.device() == k.device() && x.scalar_type() == k.scalar_type() && x.is_contiguous(),
                        "payload type/layout");
        TORCH_CHECK(sk.sizes() == torch::IntArrayRef({args.heads, args.r}) && sq.sizes() == sk.sizes() &&
                        v.sizes() == torch::IntArrayRef({args.heads, args.d}),
                    "payload shape");
        launch_gpu_planner(args, labels.data_ptr<int>(), sk.data_ptr(), sq.data_ptr(), v.data_ptr(),
                           k.scalar_type() == torch::kBFloat16, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Reused output buffer: clone if retaining an old step's result.
        return output;
    }
};
void bind_gpu_planner(pybind11::module_ &m) {
    pybind11::class_<GpuPlanner>(m, "GpuPlanner")
        .def(pybind11::init<std::vector<torch::Tensor>, int>())
        .def("step", &GpuPlanner::step, pybind11::call_guard<pybind11::gil_scoped_release>());
}
