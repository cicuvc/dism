#define main resource_probe_main
#include "resources.cu"
#undef main
#define main upstream_scan_main
#include <diagonal_scan_test.cu>
#undef main
#define main upstream_reduce_main
#include <diagonal_reduce_test.cu>
#undef main
#include <vector>

template<int C> bool dism_case(int mode) {
    constexpr int H=32,W=2*C,S=H*W;
    float* data;
    check(cudaMallocManaged(&data,2*S*sizeof(float)));
    std::vector<double> ref(2*S);
    for(int i=0;i<H;++i) for(int j=0;j<W;++j) {
        float m=0.125f*((i*17+j*7)%21-10);
        if(mode==1 && (i*3+j)%7==0) m=-INFINITY;
        if(mode==2) m=-INFINITY;
        if(mode==3 && j>i) m=-INFINITY;
        data[i*W+j]=data[S+i*W+j]=m;
        double a=0,b=-INFINITY;
        if(i && j) { a=ref[(i-1)*W+j-1]; b=ref[S+(i-1)*W+j-1]; }
        ref[i*W+j]=a+m;
        // Independent sequential double-precision recurrence.
        ref[S+i*W+j]=double(m)+std::log1p(std::exp2(b))*1.4426950408889634;
    }
    kernel<16,C,BinaryElement,LogAffine><<<1,32>>>(data);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    bool ok=true; double error=0;
    for(int i=0;i<2*S;++i) {
        if(std::isinf(ref[i])) ok &= data[i]==ref[i];
        else {
            ok &= std::isfinite(data[i]);
            error=std::max(error,std::abs(data[i]-ref[i]));
        }
    }
    ok &= error<2e-4;
    printf("Dism log-affine 16x%d mode=%d max_abs=%g [%s]\n",C,mode,error,ok?"PASS":"FAIL");
    check(cudaFree(data)); return ok;
}
int main() {
    bool ok=true;
    for(int mode=0;mode<4;++mode) {ok &= dism_case<32>(mode);ok &= dism_case<64>(mode);}
    ok &= run_reduce_case<16,32,BinaryElement,LogAffine,F32x2,false>("log-affine");
    ok &= run_reduce_case<16,64,BinaryElement,LogAffine,F32x2,false>("log-affine");
    return ok?0:1;
}
