// Check root-component membership after every extension, including clones.
#include "prefill_matcher.hpp"
#include <random>
#include <iostream>
#include <cassert>
int main(){
 std::mt19937 rng(74851);dism_prefill::PrefillMatcher sam;int clones=0;
 for(int t=0;t<200;++t){
  int n=1+rng()%128,last=0;sam.reset(n);
  for(int i=0;i<n;++i){
   int a=t%4==0?0:t%4==1?i%3:int(rng()%(t%4==2?7:512))-3;
   sam.extend(a,i,last);std::vector<int> count(sam.branch_roots.size()),seen(i+1);
   for(int v=1;v<int(sam.nodes.size());++v){
    int root=v;while(sam.nodes[root].link)root=sam.nodes[root].link;
    int g=sam.branch[v];assert(g>=0 && g<int(count.size()));assert(sam.branch_roots[g]==root);++count[g];
    if(sam.nodes[v].position<0)++clones;
   }
   for(int g=0;g<int(count.size());++g)assert(count[g]>0 && sam.branch[sam.branch_roots[g]]==g);
   for(int g=0;g<int(count.size());++g){
    int prev=-1;for(int j=sam.key_head[g];j>=0;j=sam.key_next[j]){assert(j>prev && j<=i);++seen[j];prev=j;}
    assert(prev==sam.key_tail[g]);
   }
   for(int v=1;v<int(sam.nodes.size());++v)if(sam.nodes[v].position>=0){
    int j=sam.nodes[v].position;assert(seen[j]==1);int g=sam.branch[v],found=0;
    for(int k=sam.key_head[g];k>=0;k=sam.key_next[k])found+=k==j;assert(found==1);
   }
  }
 }
 std::cout<<"200 incremental SAM branch cases passed; clone-prefix checks="<<clones<<"\n";
}
