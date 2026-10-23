// CPU-only six-layer submission probe. Reuses the diagnostic's native output
// allocation; no CUDA, pinned memory, or Python coordinator overhead.
#define main single_planner_main
#include "profile_planner_native.cpp"
#undef main
#include <algorithm>
#include <array>
int main(int argc,char** argv) {
 if(argc!=5){std::cerr<<"input.bin workers stagger_ms repeats\n";return 2;}
 std::ifstream f(argv[1],std::ios::binary);int shape[4];read(f,shape,4);
 int b=shape[0],h=shape[1],n=shape[2],nd=shape[3];
 std::vector<int> labels(size_t(2)*b*h*n);std::vector<uint8_t> hard(b*h);
 std::vector<double> tau(h);std::vector<int64_t> docs(nd*3);
 read(f,labels.data(),labels.size());read(f,hard.data(),hard.size());read(f,tau.data(),tau.size());read(f,docs.data(),docs.size());
 int workers=std::stoi(argv[2]),repeats=std::stoi(argv[4]);double stagger=std::stod(argv[3]);
 auto pool=std::make_shared<PlanningWorkers>(workers);
 for(int round=-3;round<repeats;++round){
  std::array<std::thread,6> coordinators;std::array<double,6> finish{},plan{},pack{};
  auto origin=Clock::now()+std::chrono::milliseconds(3);
  for(int layer=0;layer<6;++layer)coordinators[layer]=std::thread([&,layer]{
   std::this_thread::sleep_until(origin+std::chrono::microseconds(int64_t(layer*stagger*1000)));
   auto start=Clock::now();
   auto p=std::make_unique<TrainingBatchProgram>(pool,labels.data(),b,h,n,tau.data(),1,docs.data(),nd,hard.data(),64,true);
   plan[layer]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
   auto out=std::make_unique<Output>(*p);auto ready=Clock::now();
   p->pack_into_compact(out->buckets,out->multi);
   pack[layer]=std::chrono::duration<double,std::milli>(Clock::now()-ready).count();
   out.reset();p.reset();
   finish[layer]=std::chrono::duration<double,std::milli>(Clock::now()-origin).count();
  });
  for(auto& t:coordinators)t.join();
  if(round>=0){
   std::cout<<"{\"finish_ms\":[";for(int i=0;i<6;++i)std::cout<<(i?",":"")<<finish[i];
   std::cout<<"],\"plan_ms\":[";for(int i=0;i<6;++i)std::cout<<(i?",":"")<<plan[i];
   std::cout<<"],\"pack_ms\":[";for(int i=0;i<6;++i)std::cout<<(i?",":"")<<pack[i];std::cout<<"]}\n";
  }
 }
}
