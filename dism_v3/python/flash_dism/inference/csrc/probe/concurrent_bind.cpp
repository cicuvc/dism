// Probe-only owner transfer. Vector math uses the unchanged GPU planner kernel.
#include "../gpu_planner_params.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
extern "C" void launch_gpu_planner(GpuPlannerArgs,const int*,const void*,const void*,const void*,bool,cudaStream_t);
extern "C" void launch_concurrent_catchup(const int*,const int*,const int*,const int*,int*,int,int,int,int,int,int,cudaStream_t);

class PreparedGpuSnapshot {
    std::vector<torch::Tensor> snapshot;
    torch::Tensor marks,slots,tasks,coefficients,output;
    GpuPlannerArgs a{};
    cudaStream_t stream;
    bool activated=false;
  public:
    PreparedGpuSnapshot(std::vector<torch::Tensor> s,int cut):snapshot(std::move(s)) {
        TORCH_CHECK(snapshot.size()==10,"snapshot fields");auto &k=snapshot[0];
        TORCH_CHECK(k.is_cuda() && k.dim()==3 && (k.scalar_type()==torch::kBFloat16 || k.scalar_type()==torch::kFloat32),"CUDA BF16/FP32 key cache required");
        TORCH_CHECK(k.size(0)>0 && k.size(0)<=128 && k.size(1)>0 && k.size(1)<=(1<<20) && (k.size(2)==16||k.size(2)==32),"key dimensions outside probe bounds");
        TORCH_CHECK(snapshot[1].dim()==3 && snapshot[3].dim()==3 && snapshot[4].dim()==3 && snapshot[6].dim()==3 && snapshot[2].dim()==2,"snapshot ranks");
        c10::cuda::CUDAGuard guard(k.device());stream=at::cuda::getCurrentCUDAStream();
        a.heads=k.size(0);a.capacity=k.size(1);a.r=k.size(2);a.d=snapshot[1].size(2);
        TORCH_CHECK(cut>0 && cut<=a.capacity,"invalid cutoff");
        for(auto &x:snapshot)TORCH_CHECK(x.device()==k.device()&&x.is_contiguous(),"snapshot device/layout");
        a.snapshot=cut;a.interval=snapshot[6].size(2);a.nodes=snapshot[3].size(2);a.edge_capacity=snapshot[4].size(2);
        TORCH_CHECK(snapshot[5].scalar_type()==torch::kInt32 && snapshot[5].sizes()==torch::IntArrayRef({a.heads,8}),"state shape");
        TORCH_CHECK((a.d==32||a.d==64) && snapshot[1].scalar_type()==k.scalar_type() && snapshot[1].sizes()==torch::IntArrayRef({a.heads,a.capacity,a.d}),"value shape/dtype");
        TORCH_CHECK(a.nodes>0 && a.nodes<=2*a.capacity+2 && a.nodes%2==0 && a.edge_capacity>0 && a.edge_capacity<=3*a.capacity && a.interval>0 && a.interval<=(1<<20),"metadata bounds");
        TORCH_CHECK(snapshot[3].scalar_type()==torch::kInt32 && snapshot[3].sizes()==torch::IntArrayRef({a.heads,13,a.nodes}),"topology shape/dtype");
        TORCH_CHECK(snapshot[4].scalar_type()==torch::kInt32 && snapshot[4].sizes()==torch::IntArrayRef({a.heads,2,a.edge_capacity}),"edges shape/dtype");
        TORCH_CHECK(snapshot[6].scalar_type()==torch::kInt32 && snapshot[6].sizes()==torch::IntArrayRef({a.heads,2,a.interval}),"band shape/dtype");
        TORCH_CHECK(snapshot[7].scalar_type()==torch::kInt32 && snapshot[7].sizes()==torch::IntArrayRef({a.capacity,3,a.heads}),"history shape/dtype");
        TORCH_CHECK(snapshot[2].scalar_type()==torch::kFloat32 && snapshot[2].size(1)==a.r*a.d && snapshot[2].size(0)<=int64_t(a.heads)*4*a.capacity,"summary shape/dtype");
        TORCH_CHECK(snapshot[8].scalar_type()==torch::kFloat32 && snapshot[8].sizes()==torch::IntArrayRef({snapshot[2].size(0)}) && snapshot[9].scalar_type()==torch::kFloat32 && snapshot[9].sizes()==torch::IntArrayRef({a.heads}),"counts/tau shape/dtype");
        TORCH_CHECK(int64_t(a.capacity)+snapshot[2].size(0)<=INT_MAX/a.heads,"task address overflow");
        a.task_capacity=a.capacity+snapshot[2].size(0);
        auto io=k.options().dtype(torch::kInt32);
        marks=torch::full({a.heads,a.task_capacity},-1,io);slots=torch::empty_like(marks);tasks=torch::empty_like(marks);
        coefficients=torch::empty({a.heads,a.task_capacity},k.options().dtype(torch::kFloat32));output=torch::empty({a.heads,a.d},coefficients.options());
        a.keys=k.data_ptr();a.values=snapshot[1].data_ptr();a.summaries=snapshot[2].data_ptr<float>();
        a.topology=snapshot[3].data_ptr<int>();a.edges=snapshot[4].data_ptr<int>();a.state=snapshot[5].data_ptr<int>();
        a.band=snapshot[6].data_ptr<int>();a.history=snapshot[7].data_ptr<int>();a.counts=snapshot[8].data_ptr<float>();a.tau=snapshot[9].data_ptr<float>();
        a.marks=marks.data_ptr<int>();a.slots=slots.data_ptr<int>();a.tasks=tasks.data_ptr<int>();a.coefficients=coefficients.data_ptr<float>();a.output=output.data_ptr<float>();
    }
    void activate_and_catch_up(torch::Tensor old) {
        c10::cuda::CUDAGuard guard(snapshot[0].device());TORCH_CHECK(!activated,"snapshot already activated");
        TORCH_CHECK(old.device()==snapshot[0].device()&&old.scalar_type()==torch::kInt32&&old.is_contiguous()&&old.sizes()==torch::IntArrayRef({a.heads,8}),"old state shape/device");
        auto current=at::cuda::getCurrentCUDAStream();stream=current;activated=true;
        // Background allocations now have foreground consumers, including the
        // private task buffers that Python cannot record_stream itself.
        for(auto &x:snapshot)c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(),current);
        for(auto &x:{marks,slots,tasks,coefficients,output,old})c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(),current);
        launch_concurrent_catchup(a.topology,a.edges,a.history,old.data_ptr<int>(),a.state,a.heads,a.nodes,a.edge_capacity,a.snapshot,a.capacity,a.interval,stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    torch::Tensor step(torch::Tensor labels,torch::Tensor sk,torch::Tensor sq,torch::Tensor v) {
        auto &k=snapshot[0];c10::cuda::CUDAGuard guard(k.device());
        TORCH_CHECK(activated && at::cuda::getCurrentCUDAStream()==stream,"inactive snapshot or stream mismatch");
        TORCH_CHECK(labels.device()==k.device()&&labels.scalar_type()==torch::kInt32&&labels.is_contiguous()&&labels.sizes()==torch::IntArrayRef({3,a.heads}),"labels must be int32 [3,H]");
        for(auto &x:{sk,sq,v})TORCH_CHECK(x.device()==k.device()&&x.scalar_type()==k.scalar_type()&&x.is_contiguous(),"payload dtype/device/layout");
        TORCH_CHECK(sk.sizes()==torch::IntArrayRef({a.heads,a.r})&&sq.sizes()==sk.sizes()&&v.sizes()==torch::IntArrayRef({a.heads,a.d}),"payload shape");
        launch_gpu_planner(a,labels.data_ptr<int>(),sk.data_ptr(),sq.data_ptr(),v.data_ptr(),k.scalar_type()==torch::kBFloat16,stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
    }
};
void bind_concurrent_probe(pybind11::module_ &m) {
    pybind11::class_<PreparedGpuSnapshot>(m,"PreparedGpuSnapshot")
      .def(pybind11::init<std::vector<torch::Tensor>,int>(),pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("activate_and_catch_up",&PreparedGpuSnapshot::activate_and_catch_up,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("step",&PreparedGpuSnapshot::step,pybind11::call_guard<pybind11::gil_scoped_release>());
}
