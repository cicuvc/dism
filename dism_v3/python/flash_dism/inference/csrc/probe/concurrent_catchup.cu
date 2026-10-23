// Query-only handoff into a newly built frozen SAM. No Torch dependency.
#include <cuda_runtime.h>
__device__ __forceinline__ int lookup(const int *t,const int *e,int nodes,int edges,int u,int symbol) {
    int lo=t[3*nodes+u],hi=t[4*nodes+u],end=hi;
    while(lo<hi){int m=(lo+hi)/2;if(e[m]<symbol)lo=m+1;else hi=m;}
    return lo<end && e[lo]==symbol?e[edges+lo]:-1;
}
__global__ void catchup(const int *topology,const int *edges,const int *history,
                       const int *old_state,int *state,int heads,int nodes,int edge_pitch,
                       int cut,int capacity,int interval) {
    int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=heads)return;
    const int *old=old_state+8*h;int *s=state+8*h;
    int end=old[0];
    if(old[6] || end<cut || end>capacity || end-cut>interval || s[0]!=cut) {
        s[6]=old[6]?old[6]:2;return;
    }
    const int *t=topology+int64_t(h)*13*nodes,*e=edges+int64_t(h)*2*edge_pitch;
    int u=s[1],length=s[2];
    for(int i=cut;i<end;++i) {
        int reset=history[(int64_t(i)*3+2)*heads+h];
        if(reset!=0 && reset!=1){s[6]=3;return;}
        if(reset)u=length=0;
        int symbol=history[(int64_t(i)*3+1)*heads+h],next=lookup(t,e,nodes,edge_pitch,u,symbol);
        while(u && next<0){u=t[u];length=min(length,t[nodes+u]);next=lookup(t,e,nodes,edge_pitch,u,symbol);}
        if(next<0)u=length=0;else {u=next;++length;}
    }
    s[0]=end;s[1]=u;s[2]=length;s[3]=s[4]=-1;s[5]=0;s[6]=0;s[7]=0;
}
extern "C" void launch_concurrent_catchup(const int *t,const int *e,const int *history,
                                         const int *old,int *state,int h,int n,int ep,
                                         int cut,int cap,int interval,cudaStream_t stream) {
    catchup<<<(h+127)/128,128,0,stream>>>(t,e,history,old,state,h,n,ep,cut,cap,interval);
}
