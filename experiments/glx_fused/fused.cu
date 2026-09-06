// Bounded integration probe, not a production voc_dism implementation.
#define main tma_probe_main
#include "../glx_tma_permute/tma_mma_permute.cu"
#undef main
#define main resource_probe_main
#include "../glx_scan/resources.cu"
#undef main

// Experimental row counter, NOT the final PyTorch generator contract.
__host__ __device__ unsigned row_random(unsigned row, unsigned seed) {
    unsigned x = row ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}
__host__ __device__ float modify(float dot, int row, int col, int n, int mode) {
    if(col >= n || col > row) return -INFINITY;
    bool hard = mode == 1 || mode == 3 || (mode == 2 && (row_random(row, 123) & 1));
    if(hard) return mode != 3 && row % 7 == col % 7 ? 0.5f : -INFINITY;
    // Deliberately increasing tile maxima, plus column metadata before roll.
    return dot * 0.015625f - 2.f + (col / 64) * 2.f - (col % 11) * 0.03125f;
}

__device__ void wait_phase(uint64_t* bar, int phase) {
    unsigned a = shared_address(bar);
    asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }"
                 :: "r"(a), "r"(phase) : "memory");
}

template<int D, int DV>
__global__ void fused(const __nv_bfloat16* q,
        __grid_constant__ const CUtensorMap km,
        __grid_constant__ const CUtensorMap vm, float* output, int n, int mode) {
    constexpr int C = 64;
    using Buf = glx::MMABuffer<16,C,glx::BinaryElement,LogAffine,glx::F32x2>;
    extern __shared__ __align__(128) unsigned char sm[];
    auto& qs = *reinterpret_cast<kt::st_bf<16,D>*>(sm);
    auto& ks = *reinterpret_cast<kt::st_bf<C,D>*>(sm + sizeof(qs));
    auto& vs = *reinterpret_cast<kt::st_bf<C,DV>*>(sm + sizeof(qs) + sizeof(ks));
    auto* bar = reinterpret_cast<uint64_t*>(sm + sizeof(qs) + sizeof(ks) + sizeof(vs));
    int lane = threadIdx.x;
    if(lane == 0) {
        initialize_barrier(bar);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    for(int i=lane;i<16*D;i+=32) qs[int2{i/D,i%D}] = q[i];
    __syncthreads();
    typename Buf::VState left;
    kt::rt_fl<16,DV> out{0.f};
    float maximum[2]{0,0}, denom[2]{1,1};
    for(int tile=0;tile<3;++tile) {
        if(lane==0) {
            expect_bytes(bar,sizeof(ks)+sizeof(vs));
            constexpr int SK=D==32?32:64, SV=DV==32?32:64;
            #pragma unroll
            for(int s=0;s<D/SK;++s) tma_load_5d(&km,ks.data+s*C*SK,bar,0,0,0,tile*C,s);
            #pragma unroll
            for(int s=0;s<DV/SV;++s) tma_load_5d(&vm,vs.data+s*C*SV,bar,0,0,0,tile*C,s);
        }
        wait_phase(bar,tile&1);
        __syncthreads();
        glx::MMABuffer<16,C,glx::UnaryElement,glx::AddOp,glx::F32x2> scalar;
        {
            kt::rt_bf<16,D> a;
            kt::rt_bf<C,D> b;
            kt::rt_fl<16,C> acc{0.f};
            kt::warp::load(a,qs); kt::warp::load(b,ks);
            kt::warp::wmma::mma_ABt(acc,a,b,acc);
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto x=acc.tiles[0][c/2].data[r+2*(c&1)];
                    auto p=Buf::layout(r,c,0), t=Buf::layout(r,c,1);
                    scalar.data[r][c].value={modify(x.x,128+p.first,tile*C+p.second,n,mode),
                                             modify(x.y,128+t.first,tile*C+t.second,n,mode)};
                }
            }
        }
        scalar.roll();
        Buf pair;
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) { auto x=scalar.data[r][c].value; pair.data[r][c]={x,x}; }
        }
        auto state=pair.inclusive_scan(left,{}); left=state.first;
        // Only W survives; unroll one scalar component, not the affine pair.
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) scalar.data[r][c].value=pair.data[r][c].second;
        }
        scalar.template roll<false>();
        kt::rt_bf<16,C> weights;
        #pragma unroll
        for(int r=0;r<2;++r) {
            float m=maximum[r];
            #pragma unroll
            for(int c=0;c<8;++c) { auto x=scalar.data[r][c].value; m=fmaxf(m,fmaxf(x.u0,x.u1)); }
            m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));
            m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
            float alpha=exp2f(maximum[r]-m), sum=0;
            #pragma unroll
            for(int c=0;c<8;++c) {
                auto x=scalar.data[r][c].value;
                float a=exp2f(x.u0-m), b=exp2f(x.u1-m); sum+=a+b;
                weights.tiles[0][c/2].data[r+2*(c&1)]=__floats2bfloat162_rn(a,b);
            }
            sum+=__shfl_xor_sync(0xffffffff,sum,1);
            sum+=__shfl_xor_sync(0xffffffff,sum,2);
            denom[r]=denom[r]*alpha+sum; maximum[r]=m;
            #pragma unroll
            for(int j=0;j<DV/16;++j) {
                #pragma unroll
                for(int k=r;k<4;k+=2) { out.tiles[0][j].data[k].x*=alpha; out.tiles[0][j].data[k].y*=alpha; }
            }
        }
        {
            kt::rt_bf<C,DV,kt::ducks::rt_layout::col> value;
            kt::warp::load(value,vs);
            kt::warp::wmma::mma_AB(out,weights,value,out);
        }
        __syncthreads(); // No next TMA overwrite before all ldmatrix reads finish.
    }
    #pragma unroll
    for(int j=0;j<DV/16;++j) {
        #pragma unroll
        for(int k=0;k<4;++k) {
            int row=(k%2)*8+lane/4, col=j*16+(k/2)*8+(lane%4)*2;
            auto x=out.tiles[0][j].data[k];
            output[row*DV+col]=x.x/denom[k%2]; output[row*DV+col+1]=x.y/denom[k%2];
        }
    }
}

template<int D,int DV> bool test_fused() {
    __nv_bfloat16 *q,*k,*v; float* out;
    check(cudaMallocManaged(&q,16*D*2)); check(cudaMallocManaged(&k,192*D*2));
    check(cudaMallocManaged(&v,192*DV*2)); check(cudaMallocManaged(&out,16*DV*4));
    for(int i=0;i<16*D;++i) q[i]=__float2bfloat16(float((i*17+i/13)%7-3));
    for(int i=0;i<192*D;++i) k[i]=__float2bfloat16(float((i*11+i/7)%7-3));
    for(int i=0;i<192*DV;++i) v[i]=__float2bfloat16(float((i*13+i/17)%31-15)/16);
    CUtensorMap km{},vm{};
    bool ok=encode_permuted_b_map<64,D>(&km,k,192) && encode_permuted_b_map<64,DV>(&vm,v,192);
    constexpr int shared=sizeof(kt::st_bf<16,D>)+sizeof(kt::st_bf<64,D>)+sizeof(kt::st_bf<64,DV>)+8;
    cudaFuncAttributes attr; check(cudaFuncGetAttributes(&attr,fused<D,DV>));
    double worst=0, worst_quantized=0;
    int rescale_events[3]{0,0,0};
    for(int n : {139,192}) for(int mode=0;mode<4;++mode) {
        fused<D,DV><<<1,32,shared>>>(q,km,vm,out,n,mode);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        // Independent double recurrence, no quantization of softmax weights.
        std::vector<double> w(16*192,0);
        for(int r=0;r<16;++r) for(int c=0;c<192;++c) {
            double dot=0;
            for(int d=0;d<D;++d) dot+=double(__bfloat162float(q[r*D+d]))*__bfloat162float(k[c*D+d]);
            double m=modify(float(dot),128+r,c,n,mode);
            w[r*192+c]=std::exp2(m)*(1+(r&&c?w[(r-1)*192+c-1]:0));
        }
        for(int r=0;r<16;++r) {
            double den=1; for(int c=0;c<192;++c) den+=w[r*192+c];
            double online_max=0, online_den=1;
            std::vector<double> online_out(DV,0);
            for(int t=0;t<3;++t) {
                double m=online_max;
                for(int c=t*64;c<(t+1)*64;++c) m=std::max(m,std::log2(w[r*192+c]));
                if(m>online_max) ++rescale_events[t];
                double alpha=std::exp2(online_max-m);
                online_den*=alpha;
                for(auto& x:online_out) x*=alpha;
                for(int c=t*64;c<(t+1)*64;++c) {
                    double p=w[r*192+c]*std::exp2(-m);
                    online_den+=p;
                    double pq=__bfloat162float(__float2bfloat16_rn(float(p)));
                    for(int d=0;d<DV;++d) online_out[d]+=pq*__bfloat162float(v[c*DV+d]);
                }
                online_max=m;
            }
            for(int d=0;d<DV;++d) {
                double ref=0; for(int c=0;c<192;++c) ref+=w[r*192+c]*__bfloat162float(v[c*DV+d]);
                double err=std::abs(out[r*DV+d]-ref/den); worst=std::max(worst,err);
                if(!std::isfinite(out[r*DV+d]) || err>0.003) ok=false;
                double quantized_error=std::abs(out[r*DV+d]-online_out[d]/online_den);
                worst_quantized=std::max(worst_quantized,quantized_error);
                if(quantized_error>3e-5) ok=false;
                if(mode==3 && out[r*DV+d]!=0) ok=false;
            }
        }
    }
    if(rescale_events[1]==0 || rescale_events[2]==0) ok=false;
    std::fprintf(stdout,"D=%d DV=%d cases=8 max_abs=%.8g quantized_oracle_abs=%.8g rescales=%d/%d/%d regs=%d local=%zu static_shared=%zu dynamic_shared=%d %s\n",
                 D,DV,worst,worst_quantized,rescale_events[0],rescale_events[1],rescale_events[2],attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,shared,ok?"PASS":"FAIL");
    check(cudaFree(q));check(cudaFree(k));check(cudaFree(v));check(cudaFree(out));return ok;
}
int main() {
    bool ok=true;
    ok &= test_fused<32,32>(); ok &= test_fused<32,64>(); ok &= test_fused<32,128>();
    ok &= test_fused<64,32>(); ok &= test_fused<64,64>(); ok &= test_fused<64,128>();
    ok &= test_fused<128,32>(); ok &= test_fused<128,64>(); ok &= test_fused<128,128>();
    return ok?0:1;
}
