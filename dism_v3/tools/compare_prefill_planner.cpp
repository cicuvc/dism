// CPU-only diagnostic: production planners, identical frozen real inputs.
// The builder/worker instrumentation and LUT ablation are isolated build copies.
#include "prefill_inference.hpp"
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
using namespace dism_prefill;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
double cpu(){timespec t{};clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
template<class T>void read(std::ifstream&f,T*p,size_t n){f.read((char*)p,n*sizeof(T));if(!f)throw std::runtime_error("truncated input");}
struct Input {
 int b,h,n,nd;std::vector<int>labels;std::vector<uint8_t>hard;std::vector<double>tau;std::vector<int64_t>docs;
 Input(const char*path){std::ifstream f(path,std::ios::binary);int s[4];read(f,s,4);b=s[0];h=s[1];n=s[2];nd=s[3];
  if(b<1||h<1||n<1||nd<1||int64_t(b)*h*n>INT32_MAX)throw std::runtime_error("shape");
  labels.resize(size_t(2)*b*h*n);hard.resize(b*h);tau.resize(h);docs.resize(nd*3);
  read(f,labels.data(),labels.size());read(f,hard.data(),hard.size());read(f,tau.data(),tau.size());read(f,docs.data(),docs.size());}
};
struct Output {
 std::vector<std::unique_ptr<int[]>>ints;std::vector<std::unique_ptr<float[]>>floats;
 std::vector<std::unique_ptr<int64_t[]>>longs;size_t bytes=0;
 TrainingBuffers data{};std::vector<CompactBucketBuffers>buckets;PrefillRoutingBuffers routing{};
 int* integer(size_t n){bytes+=4*n;ints.emplace_back(new int[n]);return ints.back().get();}
 float* real(size_t n){bytes+=4*n;floats.emplace_back(new float[n]);return floats.back().get();}
 int64_t* index(size_t n){bytes+=8*n;longs.emplace_back(new int64_t[n]);return longs.back().get();}
 Output(const TrainingBatchProgram&p,const PrefillBatchProgram*inf,bool gradients){
  if(p.compact)for(int i=0;i<3;++i){size_t blocks=p.bucket_blocks[i],slots=blocks*compact_widths[i];CompactBucketBuffers v;
   v.rows=integer(slots);v.prefixes=real(slots);v.weights=real(slots);v.dlog=gradients?real(slots):nullptr;
   v.lengths=integer(blocks);v.heads=integer(blocks);buckets.push_back(v);}
  size_t streams=p.stream_base.back(),chunks=p.chunk_base.back(),slots=chunks*p.width;
  data.offsets=index(streams+1);data.rows=integer(slots);data.prefixes=real(slots);data.weights=real(slots);
  data.dlog=gradients?real(slots):nullptr;data.lengths=integer(chunks);data.resets=integer(chunks);
  data.stream_heads=inf?nullptr:integer(streams);
  if(inf){size_t nlong=inf->long_base.back();routing.stream_ids=index(streams);routing.chunk_ids=index(chunks);
   routing.local_offsets=index(streams);routing.short_stream_ids=index(inf->short_base.back());routing.long_stream_ids=index(nlong);
   routing.long_chunk_ids=index(inf->long_chunk_base.back());routing.long_local_offsets=index(nlong);}
 }
};
uint64_t hash=1469598103934665603ULL;
std::ofstream dump;
template<class T>void digest(const std::vector<T>&v){
 if(dump.is_open()){uint64_t size=v.size()*sizeof(T);dump.write((const char*)&size,sizeof(size));dump.write((const char*)v.data(),size);if(!dump)throw std::runtime_error("program dump failed");}
 const uint8_t*p=(const uint8_t*)v.data();for(size_t i=0;i<v.size()*sizeof(T);++i){hash^=p[i];hash*=1099511628211ULL;}}
int main(int argc,char**argv){
 if(argc<6){std::cerr<<"workers mode repeats warmup input.bin ...\n";return 2;}
 int workers=std::stoi(argv[1]),repeats=std::stoi(argv[3]),warmup=std::stoi(argv[4]);std::string mode=argv[2];
 if(workers<1||repeats<1||warmup<1)throw std::runtime_error("positive workers/repeats/warmup required");
 if(const char*path=std::getenv("DISM_PLANNER_PROGRAM_DUMP")){dump.open(path,std::ios::binary);if(!dump)throw std::runtime_error("cannot open program dump");}
 bool gradients=mode=="training16"||mode=="training64compact";
 bool compact=mode=="training64compact"||mode=="inference64compact";
 bool prefill=mode=="prefill16"||mode=="prefill64";
 int width=(mode.find("64")!=std::string::npos)?64:16;
 if(!prefill&&!gradients&&mode!="inference64compact"&&mode!="bare16")throw std::runtime_error("mode");
 std::vector<Input>inputs;for(int i=5;i<argc;++i)inputs.emplace_back(argv[i]);
 auto pool=std::make_shared<PlanningWorkers>(workers);
 std::vector<uint64_t>hashes;
 std::cout<<std::setprecision(10)<<"{\"workers\":"<<workers<<",\"mode\":\""<<mode<<"\",\"samples\":[";
 for(int iteration=-warmup;iteration<repeats;++iteration){
  if(iteration>=0){if(iteration)std::cout<<",";std::cout<<"[";}
  for(size_t i=0;i<inputs.size();++i){auto&x=inputs[i];double c0=cpu();auto t0=Clock::now();
   std::unique_ptr<PrefillBatchProgram>inf;std::shared_ptr<TrainingBatchProgram>p;
   if(prefill){inf=std::make_unique<PrefillBatchProgram>(pool,x.labels.data(),x.b,x.h,x.n,x.tau.data(),1,nullptr,width,64);p=inf->program;}
   else p=std::make_shared<TrainingBatchProgram>(pool,x.labels.data(),x.b,x.h,x.n,x.tau.data(),1,x.docs.data(),x.nd,x.hard.data(),width,compact,gradients);
   auto t1=Clock::now();double c1=cpu();auto out=std::make_unique<Output>(*p,inf.get(),gradients);auto t2=Clock::now();
   if(inf)inf->pack_into(out->data,out->routing);else if(compact)p->pack_into_compact(out->buckets,out->data);else p->pack_into(out->data);
   auto t3=Clock::now();double c3=cpu();size_t bytes=out->bytes,events=p->events;
   auto job_ms=p->diagnostic_job_ms;
   if(iteration==-warmup){hash=1469598103934665603ULL;for(const auto&program:p->programs){digest(program.offsets);digest(program.rows);digest(program.decay);digest(program.weight);digest(program.logden);}hashes.push_back(hash);}
   auto t4=Clock::now();out.reset();p.reset();inf.reset();auto t5=Clock::now();double c5=cpu();
   if(iteration>=0){if(i)std::cout<<",";std::cout<<"{\"plan_ms\":"<<ms(t0,t1)<<",\"plan_cpu_ms\":"<<c1-c0<<",\"allocate_ms\":"<<ms(t1,t2)
    <<",\"pack_ms\":"<<ms(t2,t3)<<",\"release_ms\":"<<ms(t4,t5)<<",\"total_ms\":"<<ms(t0,t3)+ms(t4,t5)
    <<",\"cpu_ms\":"<<c3-c0+c5-c3<<",\"metadata_bytes\":"<<bytes<<",\"events\":"<<events<<",\"job_ms\":[";
    for(size_t j=0;j<job_ms.size();++j){if(j)std::cout<<",";std::cout<<job_ms[j];}std::cout<<"]}";}
  }
  if(iteration>=0)std::cout<<"]";
 }
 std::cout<<"],\"forward_program_hashes\":[";for(size_t i=0;i<hashes.size();++i){if(i)std::cout<<",";std::cout<<"\""<<std::hex<<hashes[i]<<std::dec<<"\"";}std::cout<<"]}\n";
}
