#include "common/graph_route.cuh"
#include "backward/hard.cuh"

namespace DISM_VARIANT {
namespace hb=dism_hard_backward;

// Natural-tau derivative of log W_l; the near-zero series avoids cancellation.
__global__ void hard_dlog_weights_kernel(const float* logw,float* result,int stride,int heads,const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    int l=blockIdx.x*blockDim.x+threadIdx.x,head=blockIdx.y;
    if(l>=stride) return;
    float t=logw[int64_t(head)*stride+1]*0.6931471805599453f;
    float a=fabsf(t),length=float(l),x=length*t,g=0.f;
    if(l && fabsf(x)<.1f) {
        // (L+1)/2 + (L^2-1)t/12 - (L^4-1)t^3/720 + O(L^6 t^5)
        g=(length+1.f)*.5f+(length*length-1.f)*t*(1.f/12.f-(x*x+t*t)*(1.f/720.f));
    } else if(l) {
        float tail=length*a>80.f?0.f:__fdividef(length,expm1f(length*a));
        float negative=__fdividef(1.f,-expm1f(-a))-tail;
        g=t>0?length+1.f-negative:negative;
    }
    result[int64_t(head)*stride+l]=g;
}

template<class Configuration=ActiveConfig>
__global__ __launch_bounds__(384,1) void hard_backward_kernel(const __grid_constant__ hb::Args args, const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    using namespace dism_backward;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid(),lane=kt::warp::laneid();
    MbarrierRingPipe input(allocator,BufferSet{&hb::Shared::input,&hb::Shared::Labels::input},group,
                          ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    MbarrierRingPipe held(allocator,BufferSet{&hb::Shared::held,&hb::Shared::Labels::held},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    auto& shared=allocator.allocate<hb::Shared>();
    hb::Scheduler scheduler(args);
    if(group==2) hb::produce(args,shared,held,input,scheduler);
    else {
        kt::warpgroup::increase_registers<232>();
        hb::Scheduler::Task task;
        while(scheduler.next(task)) {
            if(!args.hard[task.hard_index]) continue;
            int k0=task.k0+16*(2*warp+group);
            int global_k0=task.begin+k0;
            auto first=held.waitBuffer(2,shared.held,shared.labels.held);
            kt::rt_bf<K,R> soft_keys;
            kt::rt_bf<K,DV> values;
            kt::group<8>::load(soft_keys,first.template get<0>().sk);
            kt::group<8>::load(values,first.template get<0>().v);
            kt::warpgroup::sync(group+1);
            kt::rt_fl<K,R> dsk{0.f};
            kt::rt_fl<K,DV> dv{0.f};
            for(int q0=task.n-Q;q0>=task.k0;q0-=Q) {
                auto packet=input.waitBuffer(2,shared.input,shared.labels.input);
                auto& data=packet.template get<0>();
                auto scan=hb::recompute(args,packet.template get<1>().value,first.template get<1>().value,
                                        task,task.k0,k0,q0);
                kt::rt_fl<K,Q> a{0.f},b{0.f};
                {
                    kt::rt_bf<Q,R> queries;
                    kt::warp::load<true>(queries,data.sq.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(a,soft_keys,queries,a);
                }
                {
                    kt::rt_bf<Q,DV> derivative;
                    kt::warp::load<true>(derivative,data.dout.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(b,values,derivative,b);
                }
                kt::rt_bf<K,Q> pa,pb;
                // Center g by g_(query+1), including the zero-valued fallback
                // in the final mean. Accumulate mass too: saved log2 Z has
                // FP32 rounding, which otherwise amplifies cancellation by L.
                // Reduce keys in registers; only global FP32 atomics are used.
#pragma unroll
                for(int c=0;c<4;++c) {
                    float u[2]={0.f,0.f},v[2]={0.f,0.f},w[2]={0.f,0.f},z[2]={0.f,0.f};
#pragma unroll
                    for(int r=0;r<2;++r) {
                        auto pos=Scalar::layout(r,c,0);
                        auto av=a.tiles[0][c/2].data[r+2*(c&1)];
                        auto bv=b.tiles[0][c/2].data[r+2*(c&1)];
                        float p[2];
#pragma unroll
                        for(int e=0;e<2;++e) {
                            int query=q0+pos.second+4*e;
                            int key=k0+pos.first;
                            int l=key<=query?key-(e?scan.data[r][c].value.u1:scan.data[r][c].value.u0):0;
                            int64_t off=int64_t(task.head)*args.lut_stride+l;
                            float norm=args.normalizer[
                                int64_t(task.bh)*args.score.n+task.begin+query];
                            p[e]=l?exp2_ftz(args.log_weights[off]-norm):0.f;
                            float g=args.dlog_weights[off]-args.dlog_weights[
                                int64_t(task.head)*args.lut_stride+query+1];
                            float pse=p[e]*(e?av.y:av.x)*(e?bv.y:bv.x);
                            u[e]+=pse; v[e]=fmaf(p[e],g,v[e]); w[e]=fmaf(pse,g,w[e]);
                            z[e]+=p[e];
                        }
                        pa.tiles[0][c/2].data[r+2*(c&1)]=__floats2bfloat162_rn(p[0]*av.x,p[1]*av.y);
                        pb.tiles[0][c/2].data[r+2*(c&1)]=__floats2bfloat162_rn(p[0]*bv.x,p[1]*bv.y);
                    }
#pragma unroll
                    for(int e=0;e<2;++e) {
#pragma unroll
                        for(int shift=4;shift<=16;shift*=2) {
                            u[e]+=__shfl_xor_sync(0xffffffff,u[e],shift);
                            v[e]+=__shfl_xor_sync(0xffffffff,v[e],shift);
                            w[e]+=__shfl_xor_sync(0xffffffff,w[e],shift);
                            z[e]+=__shfl_xor_sync(0xffffffff,z[e],shift);
                        }
                        if(lane<4) {
                            int query=q0+Scalar::layout(0,c,0).second+4*e;
                            int owner=kt::warpid(),col=query-q0;
                            shared.moments[owner][0][col]=u[e];
                            shared.moments[owner][1][col]=v[e];
                            shared.moments[owner][2][col]=w[e];
                            shared.moments[owner][3][col]=z[e];
                        }
                    }
                }
                {
                    kt::rt_bf<Q,DV,kt::ducks::rt_layout::col> derivative;
                    kt::warp::load(derivative,data.dout.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(dv,pa,derivative,dv);
                }
                {
                    kt::rt_bf<Q,R,kt::ducks::rt_layout::col> queries;
                    kt::warp::load(queries,data.sq.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(dsk,pb,queries,dsk);
                }
                packet.submitToNextAndTrigger(); input.moveNext();
                kt::group<8>::sync(4);
                if(kt::warpid()==0) {
                    int64_t off=int64_t(task.bh)*args.score.n+task.begin+q0+lane;
                    int64_t size=int64_t(args.score.batch)*args.score.heads*args.score.n;
#pragma unroll
                    for(int m=0;m<4;++m) {
                        float sum=0.f;
#pragma unroll
                        for(int source=0;source<8;++source) sum+=shared.moments[source][m][lane];
                        atomicAdd(args.moments+m*size+off,sum);
                    }
                }
                // query_gradient's rendezvous also protects moments from reuse.
                query_gradient(pb,first.template get<0>().sk,shared.transpose,args.score,task.bh,task.begin+q0,args,shared);
            }
            first.submitToNextAndTrigger(); held.moveNext();
            store_key_gradient<DV,0,false>(dv,args.dv,args.score,task.bh,global_k0,args,shared);
            store_key_gradient<R,1,false>(dsk,args.dsk,args.score,task.bh,global_k0,args,shared);
        }
    }
    if(group<2) kt::tma::store_async_wait<0>();
    kt::warpgroup::sync(group+1);
}

__global__ void hard_tau_reduce_kernel(const float* moments,const float* norm,const float* dlogw,
        float* output,int batch,int heads,int n,int lut_stride,
        const bool* hard,const hb::Task* tasks,int task_count,const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    int head=blockIdx.x;
    int64_t size=int64_t(batch)*heads*n;
    float sum=0.f;
    if(tasks) {
        for(int task_index=0;task_index<task_count;++task_index) {
            auto task=tasks[task_index];
            if(task.head!=head || task.k0!=0 || !hard[task.hard_index]) continue;
            for(int local=threadIdx.x;local<task.n;local+=blockDim.x) {
                int64_t off=int64_t(task.bh)*n+task.begin+local;
                float fallback=dism_backward::exp2_ftz(-norm[off]);
                float mass=moments[3*size+off]+fallback;
                float center=dlogw[int64_t(head)*lut_stride+local+1];
                float inverse=__fdividef(1.f,mass);
                float mean=fmaf(-fallback,center,moments[size+off])*inverse;
                sum+=fmaf(-moments[off],mean,moments[2*size+off])*inverse;
            }
        }
    } else {
        for(int i=threadIdx.x;i<batch*n;i+=blockDim.x) {
            int64_t off=(int64_t(i/n)*heads+head)*n+i%n;
            float fallback=dism_backward::exp2_ftz(-norm[off]);
            float mass=moments[3*size+off]+fallback;
            float center=dlogw[int64_t(head)*lut_stride+i%n+1];
            float inverse=__fdividef(1.f,mass);
            float mean=fmaf(-fallback,center,moments[size+off])*inverse;
            sum+=fmaf(-moments[off],mean,moments[2*size+off])*inverse;
        }
    }
    for(int shift=16;shift;shift/=2) sum+=__shfl_down_sync(0xffffffff,sum,shift);
    __shared__ float partials[8];
    if((threadIdx.x&31)==0) partials[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x<32) {
        sum=threadIdx.x<8?partials[threadIdx.x]:0.f;
        for(int shift=16;shift;shift/=2) sum+=__shfl_down_sync(0xffffffff,sum,shift);
        if(threadIdx.x==0) output[head]=sum;
    }
}
const void* hard_backward_address(){return reinterpret_cast<const void*>(hard_backward_kernel<>);}
const void* hard_dlog_weights_address(){return reinterpret_cast<const void*>(hard_dlog_weights_kernel);}
const void* hard_tau_reduce_address(){return reinterpret_cast<const void*>(hard_tau_reduce_kernel);}
} // namespace DISM_VARIANT
