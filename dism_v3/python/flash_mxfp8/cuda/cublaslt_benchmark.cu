// Benchmark-only, Torch-free cuBLASLt plans for row-major X @ W.T.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static thread_local std::string error;
static void check(cublasStatus_t s) {
    if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status "+std::to_string(s));
}
struct Plan {
    cublasLtHandle_t lt=nullptr;
    cublasLtMatmulDesc_t op=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,d=nullptr;
    cublasLtMatmulPreference_t pref=nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> algos;
    void *x,*w,*out,*workspace;
    size_t workspace_bytes;
    ~Plan() {
        if(pref)cublasLtMatmulPreferenceDestroy(pref);
        if(d)cublasLtMatrixLayoutDestroy(d);
        if(b)cublasLtMatrixLayoutDestroy(b);
        if(a)cublasLtMatrixLayoutDestroy(a);
        if(op)cublasLtMatmulDescDestroy(op);
        if(lt)cublasLtDestroy(lt);
    }
};
extern "C" {
const char* lt_error(){return error.c_str();}
size_t lt_version(){return cublasLtGetVersion();}
void* lt_create(void *x,void *w,void *sx,void *sw,void *out,void *workspace,
                size_t workspace_bytes,int m,int n,int k,int mode,int fast,int requested) {
    try {
        auto p=std::make_unique<Plan>();
        p->x=x;p->w=w;p->out=out;p->workspace=workspace;p->workspace_bytes=workspace_bytes;
        check(cublasLtCreate(&p->lt));
        check(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t trans=CUBLAS_OP_T,normal=CUBLAS_OP_N;
        check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
        // Column-major W.T @ X, transposed output, is row-major X @ W.T.
        cudaDataType_t type=mode==0?CUDA_R_16BF:CUDA_R_8F_E4M3;
        check(cublasLtMatrixLayoutCreate(&p->a,type,k,n,k));
        check(cublasLtMatrixLayoutCreate(&p->b,type,k,m,k));
        check(cublasLtMatrixLayoutCreate(&p->d,CUDA_R_16BF,n,m,n));
        if(mode) {
            auto scale=mode==2?CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0:CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scale,sizeof(scale)));
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scale,sizeof(scale)));
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sw,sizeof(sw)));
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sx,sizeof(sx)));
            int8_t flag=fast;
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&flag,sizeof(flag)));
        }
        check(cublasLtMatmulPreferenceCreate(&p->pref));
        check(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
        uint32_t reduction=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        check(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&reduction,sizeof(reduction)));
        std::vector<cublasLtMatmulHeuristicResult_t> found(requested);
        int count=0;
        check(cublasLtMatmulAlgoGetHeuristic(p->lt,p->op,p->a,p->b,p->d,p->d,p->pref,requested,found.data(),&count));
        for(int i=0;i<count;i++)if(found[i].state==CUBLAS_STATUS_SUCCESS)p->algos.push_back(found[i]);
        if(p->algos.empty())throw std::runtime_error("No cuBLASLt heuristic algorithms for this mode");
        return p.release();
    } catch(const std::exception& e){error=e.what();return nullptr;}
}
int lt_count(Plan *p){return p->algos.size();}
int lt_launch(Plan *p,int i,cudaStream_t stream) {
    try {
        float alpha=1,beta=0;
        check(cublasLtMatmul(p->lt,p->op,&alpha,p->w,p->a,p->x,p->b,&beta,p->out,p->d,p->out,p->d,
                            &p->algos.at(i).algo,p->workspace,p->workspace_bytes,stream));
        return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
void lt_info(Plan *p,int i,int *info,size_t *ws) {
    auto &a=p->algos.at(i);
    cublasLtMatmulAlgoConfigAttributes_t attrs[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME};
    for(int j=0;j<5;j++){size_t written=0;info[j]=-1;cublasLtMatmulAlgoConfigGetAttribute(&a.algo,attrs[j],info+j,sizeof(int),&written);}
    uint64_t flags=0;size_t written=0;
    info[6]=cublasLtMatmulAlgoCapGetAttribute(&a.algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written);
    info[5]=static_cast<int>(flags);
    *ws=a.workspaceSize;
}
void lt_destroy(Plan *p){delete p;}
}
