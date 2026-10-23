// Opt-in research prototype. No Torch headers; integer scheduling atomics only.
#include "summary_program.h"
#include <cuda/atomic>
#include <cuda_bf16.h>
__device__ float read_value(float x) { return x; }
__device__ float read_value(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void compensated_add(float x,float &sum,float &correction) {
    float y=x-correction,t=sum+y;correction=(t-sum)-y;sum=t;
}
using Atomic = cuda::atomic_ref<int,cuda::thread_scope_device>;

__device__ void push(SummaryProgramArgs a,int op,int h,int tile,int index) {
    int slot=Atomic(a.control[0]).fetch_add(1,cuda::memory_order_relaxed);
    // Queue capacity is the exact DAG task count, validated by the host compiler.
    a.queue[4*slot]=op;a.queue[4*slot+1]=h;a.queue[4*slot+2]=tile;a.queue[4*slot+3]=index;
    Atomic(a.ready[slot]).store(1,cuda::memory_order_release);
}
__global__ void reset_program(SummaryProgramArgs a) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<a.queue_capacity)a.ready[i]=i<a.initial_tasks;
    if(i<3+3*a.head_count*a.tiles)a.control[i]=i==0?a.initial_tasks:0;
}

template<class T> __global__ void interpret(SummaryProgramArgs a) {
    // Necessary inter-warp dispatch communication; vector arithmetic stays in registers.
    __shared__ int command[4];
    int lane=threadIdx.x,channels=a.r*a.d;
    for(;;) {
        if(lane==0) {
            for(;;) {
                int cursor=Atomic(a.control[1]).load(cuda::memory_order_relaxed);
                int tail=Atomic(a.control[0]).load(cuda::memory_order_relaxed);
                if(cursor<tail && Atomic(a.control[1]).compare_exchange_weak(cursor,cursor+1,cuda::memory_order_relaxed)) {
                    while(!Atomic(a.ready[cursor]).load(cuda::memory_order_acquire)) {}
                    for(int k=0;k<4;++k)command[k]=a.queue[4*cursor+k];
                    break;
                }
                if(Atomic(a.control[2]).load(cuda::memory_order_acquire)==a.jobs) {command[0]=-1;break;}
                __nanosleep(64);
            }
        }
        __syncthreads();
        int op=command[0];if(op<0)return;
        int h=command[1],tile=command[2],index=command[3];
        const int *hd=a.heads+10*h;
        int base=hd[0],n=hd[1],ebase=hd[2],blocks=hd[3],cbase=hd[4],chains=hd[5];
        int coord=tile*a.tile+lane,job=h*a.tiles+tile;
        if(coord<channels) {
            if(op==0) { // local Euler prefix, sqrt(N) nodes per instruction
                int begin=index*a.block_size,end=min(n,begin+a.block_size);float sum=0.f,correction=0.f;
                for(int i=begin;i<end;++i) {
                    int node=a.nodes[7*(base+i)+1]; // Euler order stores global node IDs
                    int p=a.nodes[7*node];
                    if(p>=0)compensated_add(read_value(((const T*)a.keys)[(int64_t(h)*a.capacity+p)*a.r+coord/a.d])*
                                     read_value(((const T*)a.values)[(int64_t(h)*a.capacity+p)*a.d+coord%a.d]),sum,correction);
                    a.prefix[int64_t(base+i)*channels+coord]=sum;
                }
                a.block_carry[int64_t(ebase+index)*channels+coord]=sum;
            } else if(op==1) { // one sequential scan of sqrt(N) block totals
                float sum=0.f,correction=0.f;
                for(int b=0;b<blocks;++b) {
                    int64_t at=int64_t(ebase+b)*channels+coord;
                    float value=a.block_carry[at];a.block_carry[at]=sum;compensated_add(value,sum,correction);
                }
            } else if(op==2) { // independent bounded chain program
                float acc=0.f,correction=0.f;
                for(int j=a.chain_offsets[index];j<a.chain_offsets[index+1];++j) {
                    int u=a.chain_nodes[j],tin=a.nodes[7*u+2],tout=a.nodes[7*u+3]-1;
                    float right=a.prefix[int64_t(base+tout)*channels+coord]+a.block_carry[int64_t(ebase+tout/a.block_size)*channels+coord];
                    float left=tin? a.prefix[int64_t(base+tin-1)*channels+coord]+a.block_carry[int64_t(ebase+(tin-1)/a.block_size)*channels+coord]:0.f;
                    float subtree=right-left;
                    int mid=a.nodes[7*u+4],sid=a.nodes[7*u+5];
                    if(mid>=0)a.output[int64_t(mid)*channels+coord]=subtree;
                    acc*=a.rho[u];correction*=a.rho[u];
                    compensated_add(a.weights[u]*subtree,acc,correction);
                    if(sid>=0)a.output[int64_t(sid)*channels+coord]=acc;
                    int boundary=a.nodes[7*u+6];
                    if(boundary>=0)a.chain_tail[int64_t(boundary)*channels+coord]=acc;
                }
            } else { // no recursive tree scan: read compressed ancestor-block list
                float carry=0.f,correction=0.f;
                for(int j=a.ancestor_offsets[index];j<a.ancestor_offsets[index+1];++j)
                    compensated_add(a.ancestor_scale[j]*a.chain_tail[int64_t(a.ancestors[j])*channels+coord],carry,correction);
                for(int j=a.chain_offsets[index];j<a.chain_offsets[index+1];++j) {
                    int u=a.chain_nodes[j],sid=a.nodes[7*u+5];
                    if(sid>=0) {
                        int64_t at=int64_t(sid)*channels+coord;
                        a.output[at]=fmaf(a.chain_scale[u],carry,a.output[at]);
                    }
                }
            }
        }
        // Publish all lanes' vector writes before the CTA's completion token.
        __threadfence();__syncthreads();
        if(lane==0) {
            if(op==0) {
                int last=Atomic(a.control[3+job]).fetch_add(1,cuda::memory_order_acq_rel);
                if(last+1==blocks)push(a,1,h,tile,0);
            } else if(op==1) {
                for(int b=0;b<chains;++b)push(a,2,h,tile,cbase+b);
            } else if(op==2) {
                int last=Atomic(a.control[3+a.head_count*a.tiles+job]).fetch_add(1,cuda::memory_order_acq_rel);
                if(last+1==chains) {
                    if(hd[8])for(int b=0;b<chains;++b)push(a,3,h,tile,cbase+b);
                    else Atomic(a.control[2]).fetch_add(1,cuda::memory_order_release);
                }
            } else {
                int last=Atomic(a.control[3+2*a.head_count*a.tiles+job]).fetch_add(1,cuda::memory_order_acq_rel);
                if(last+1==chains)Atomic(a.control[2]).fetch_add(1,cuda::memory_order_release);
            }
        }
        __syncthreads();
    }
}
extern "C" void launch_summary_program(SummaryProgramArgs a,const int *initial,bool bf16,int workers,cudaStream_t stream) {
    if(!a.jobs)return;
    int count=max(a.queue_capacity,3+3*a.head_count*a.tiles);
    reset_program<<<(count+255)/256,256,0,stream>>>(a);
    cudaMemcpyAsync(a.queue,initial,int64_t(a.initial_tasks)*4*sizeof(int),cudaMemcpyDeviceToDevice,stream);
    if(bf16)interpret<__nv_bfloat16><<<workers,a.tile,0,stream>>>(a);
    else interpret<float><<<workers,a.tile,0,stream>>>(a);
}
