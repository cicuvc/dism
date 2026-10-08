#include "varlen/probe.cuh"
#include "varlen/layout.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
template<int Rows>
__global__ void descriptor_probe(const ProbeRecord<Rows>* records,const int3* tasks) {
    __shared__ typename ProbeRecord<Rows>::Tile tile;
    __shared__ kt::semaphore ready;
    int3 task=tasks[blockIdx.x]; // sequence-record, head, local token start
    const auto& record=records[task.x];
    bool leader=kt::warp::elect_leader();
    if (leader) kt::init_semaphore(ready,1);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (leader) {
        kt::tma::expect_bytes<true>(ready,sizeof(tile));
        kt::tma::load_async(tile,record.input,{0,task.z,task.y,0},ready);
    }
    kt::wait(ready,0);
    for (int i=threadIdx.x;i<Rows*64;i+=32) {
        int physical_row=i/64,channel=i%64;
        int logical_row=(physical_row%8)*(Rows/8)+physical_row/8;
        record.output[(int64_t(task.z+logical_row)*record.heads+task.y)*64+channel]=
            tile.payload[int2{physical_row,channel}];
    }
}

template<int Rows> const void* descriptor_address() { return reinterpret_cast<const void*>(descriptor_probe<Rows>); }
template const void* descriptor_address<32>();
template const void* descriptor_address<64>();
} // namespace dism_varlen
} // namespace DISM_VARIANT
