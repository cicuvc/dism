// Optional DISM_BASELINE_PREFILL_HEADER compares against an isolated old source snapshot.
#ifdef DISM_BASELINE_PREFILL_HEADER
#define dism_decode baseline_decode
#define dism_prefill baseline_prefill
#include DISM_BASELINE_PREFILL_HEADER
#undef dism_decode
#undef dism_prefill
#include "prefill.hpp"
#else
#include "prefill.hpp"
namespace baseline_prefill=dism_prefill;
#endif
#include <cstring>
#include <iostream>
#include <random>
#include <climits>
template<class A,class B> void equal(const A& a,const B& b) {
 if(a.size()!=b.size() || (a.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))))throw std::runtime_error("byte mismatch");
}
int main(){
 std::mt19937 rng(231);dism_prefill::Builder reuse;
 for(int trial=0;trial<1400;++trial){
  int n=trial<1000?1+rng()%128:1+rng()%2048;
  std::vector<int> q(n),k(n),reset(n);
  for(int j=0;j<n;++j){int symbols=1+(trial%7==0?INT_MAX/2:trial%5==0?3:512);
   q[j]=int(rng()%symbols)-symbols/2;k[j]=int(rng()%symbols)-symbols/2;
   if(trial%11==0)q[j]=k[j]=j%7;if(trial%17==0)q[j]=k[j]=INT_MIN;
   reset[j]=rng()%32==0;
  }
  double tau=std::vector<double>{-16.,-2.,-1e-8,0.,1e-8,.5,16.}[(trial/8)%7];
  bool grad=trial%3!=0,ref=trial%4==0;
  auto expected=baseline_prefill::Builder(q,k,reset,tau,true,grad).finish();
  reuse.build(q.data(),k.data(),reset.data(),n,tau,ref,grad);auto got=reuse.finish();
  if(got.n!=expected.n ||got.states!=expected.states)throw std::runtime_error("header mismatch");
  equal(got.offsets,expected.offsets);equal(got.rows,expected.rows);equal(got.decay,expected.decay);
  equal(got.weight,expected.weight);equal(got.logden,expected.logden);equal(got.dlogweight,expected.dlogweight);equal(got.dlogden,expected.dlogden);
  for(int width:{16,32,64}){auto a=baseline_prefill::chunk_program(expected,width);auto b=dism_prefill::chunk_program(got,width);
   equal(a.offsets,b.offsets);equal(a.rows,b.rows);equal(a.lengths,b.lengths);equal(a.resets,b.resets);equal(a.prefixes,b.prefixes);equal(a.weights,b.weights);equal(a.dlog,b.dlog);
  }
 }
 std::cout<<"1400 reused-builder cases byte identical, including reference LCA, gradients, resets, tau and length changes\n";
}
