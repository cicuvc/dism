#pragma once

namespace dism_graph_route {
inline const uint32_t* validate(const std::optional<at::Tensor>& choice, c10::Device device) {
    if (!choice) return nullptr;
    TORCH_CHECK(choice->device()==device && choice->scalar_type()==at::kInt &&
                choice->numel()==1 && choice->is_contiguous(),
                "graph choice must be one contiguous CUDA int32 on the operand device");
    return reinterpret_cast<const uint32_t*>(choice->data_ptr<int>());
}
}
