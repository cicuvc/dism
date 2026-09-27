// All eight compute warps own distinct vocab rows and both output gradients.
// No P/G mailbox: only the common producer/input-ring ready/free protocol.
template<int D,int T,bool SHARED> struct VocabSymmetric {
    struct Init {kt::st_bf<16,D> key[8],value[8];};
    struct Slot {kt::st_bf<T,D> x,y,u;};
    union {Init init;Slot slot[2];};
    alignas(8) uint64_t ready[2],free[2];
};
// D64: vocabulary stays immutable in shared memory throughout the scan.
// It must NOT alias the producer's mutable input ring.
template<int T> struct VocabSymmetric<64,T,true> {
    struct Init {kt::st_bf<16,64> key[8],value[8];};
    struct Slot {kt::st_bf<T,64> x,y,u;};
    Init init;
    Slot slot[2];
    alignas(8) uint64_t ready[2],free[2];
};
template<int D,int T> __device__ __forceinline__ void dot_shared_vocab(
        kt::rt_fl<16,T>& dst,kt::st_bf<16,D>& x,kt::st_bf<T,D>& y) {
    #pragma unroll
    for(int c=0;c<T/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) dst.tiles[0][c].data[r]=float2{0.f,0.f};
    }
    #pragma unroll
    for(int f=0;f<D/32;++f) {
        kt::rt_bf<16,32> xp;
        kt::rt_bf<T,32> yp;
        auto xv=x.template subtile<16,32>({0,f});
        auto yv=y.template subtile<T,32>({0,f});
        kt::warp::load(xp,xv);
        kt::warp::load(yp,yv);
        kt::warp::wmma::mma_ABt(dst,xp,yp,dst);
    }
}
template<int D,int T,bool SHARED> __global__ __launch_bounds__(384,1) void vocabulary_symmetric(
        __grid_constant__ const Args a,__grid_constant__ const CUtensorMap xm,
        __grid_constant__ const CUtensorMap ym,__grid_constant__ const CUtensorMap um) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<VocabSymmetric<D,T,SHARED>*>(mem);
    int warp=threadIdx.x/32,lane=threadIdx.x%32,h=blockIdx.y;
    int vb=blockIdx.x*128+warp*16,nt=(a.n+T-1)/T;
    if(threadIdx.x==0) {
        for(int b=0;b<2;++b) {init_bar(&s.ready[b],32);init_bar(&s.free[b],256);}
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if(warp<8) {
        stage(s.init.key[warp],a.key,int64_t(h)*a.voc+vb,a.voc-vb,lane);
        stage(s.init.value[warp],a.value,int64_t(h)*a.voc+vb,a.voc-vb,lane);
    }
    __syncthreads();
    kt::rt_bf<16,D> key,value;
    if constexpr(!SHARED) {
        if(warp<8) {kt::warp::load(key,s.init.key[warp]);kt::warp::load(value,s.init.value[warp]);}
    }
    __syncthreads();
    if(warp>=8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if(warp==8) for(int t=0;t<a.batch*nt;++t) {
            int b=t%2,rb=t%nt*T,bh=t/nt*a.heads+h;
            if(t>=2) wait(&s.free[b],(t/2-1)&1);
            auto& slot=s.slot[b];
            if(rb+T<=a.n) {
                if(lane==0) {
                    expect(&s.ready[b],3*T*D*2);
                    issue(&xm,slot.x,&s.ready[b],rb,bh);
                    issue(&ym,slot.y,&s.ready[b],rb,bh);
                    issue(&um,slot.u,&s.ready[b],rb,bh);
                } else arrive(&s.ready[b]);
            } else {
                stage(slot.x,a.x,int64_t(bh)*a.n+rb,a.n-rb,lane);
                stage(slot.y,a.y,int64_t(bh)*a.n+rb,a.n-rb,lane);
                stage(slot.u,a.packed_u,int64_t(bh)*a.n+rb,a.n-rb,lane);
                __syncwarp();arrive(&s.ready[b]);
            }
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        kt::rt_fl<16,D> dk{0.f},dv{0.f};
        for(int t=0;t<a.batch*nt;++t) {
            int b=t%2,rb=t%nt*T,bh=t/nt*a.heads+h;
            wait(&s.ready[b],(t/2)&1);
            auto& slot=s.slot[b];
            {
                kt::rt_fl<16,T> p;
                if constexpr(SHARED) dot_shared_vocab(p,s.init.key[warp],slot.x);
                else dot(p,key,slot.x);
                probability(p,a.lx2,a,bh,rb,vb,lane);
                {
                    kt::rt_bf<16,T> pb;
                    kt::warp::copy(pb,p);
                    product(dv,pb,slot.u,1.f);
                }
                {
                    kt::rt_fl<16,T> dp;
                    if constexpr(SHARED) dot_shared_vocab(dp,s.init.value[warp],slot.u);
                    else dot(dp,value,slot.u);
                    #pragma unroll
                    for(int c=0;c<T/16;++c) {
                        #pragma unroll
                        for(int r=0;r<4;++r) {
                            int col=rb+c*16+r/2*8+2*(lane%4);
                            auto& v=p.tiles[0][c].data[r];auto d=dp.tiles[0][c].data[r];
                            v.x*=d.x-(col<a.n?a.delta[int64_t(bh)*a.n+col]:0.f);
                            v.y*=d.y-(col+1<a.n?a.delta[int64_t(bh)*a.n+col+1]:0.f);
                        }
                    }
                }
                kt::rt_bf<16,T> g;
                kt::warp::copy(g,p);product(dk,g,slot.x,a.scale);
            }
            {
                kt::rt_fl<16,T> p;
                if constexpr(SHARED) dot_shared_vocab(p,s.init.value[warp],slot.y);
                else dot(p,value,slot.y);
                probability(p,a.ly2,a,bh,rb,vb,lane);
                #pragma unroll
                for(int c=0;c<T/16;++c) {
                    #pragma unroll
                    for(int r=0;r<4;++r) {
                        int col=rb+c*16+r/2*8+2*(lane%4);
                        auto& v=p.tiles[0][c].data[r];
                        v.x*=col<a.n?a.lambda[int64_t(bh)*a.n+col]:0.f;
                        v.y*=col+1<a.n?a.lambda[int64_t(bh)*a.n+col+1]:0.f;
                    }
                }
                kt::rt_bf<16,T> g;
                kt::warp::copy(g,p);product(dv,g,slot.y,a.scale);
            }
            __syncwarp();arrive(&s.free[b]);
        }
        store(dk,a.dkey,int64_t(h)*a.voc+vb,a.voc-vb,lane);
        store(dv,a.dvalue,int64_t(h)*a.voc+vb,a.voc-vb,lane);
    }
    __syncthreads();
}
