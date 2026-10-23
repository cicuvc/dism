// Independent non-sampling Linux perf counters for a child workload and threads.
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>
struct Event {const char* name;uint32_t type;uint64_t config;int fd=-1;};
int main(int argc,char**argv) {
    if(argc<2)return 2;
    pid_t child=fork();
    if(child<0)return 3;
    if(!child) {raise(SIGSTOP);execvp(argv[1],argv+1);_exit(127);}
    int status;waitpid(child,&status,WUNTRACED);
    std::vector<Event> events={
        {"cycles",PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES},
        {"instructions",PERF_TYPE_HARDWARE,PERF_COUNT_HW_INSTRUCTIONS},
        {"branches",PERF_TYPE_HARDWARE,PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {"branch_misses",PERF_TYPE_HARDWARE,PERF_COUNT_HW_BRANCH_MISSES},
        {"l1d_read_misses",PERF_TYPE_HW_CACHE,PERF_COUNT_HW_CACHE_L1D|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16)},
        {"llc_read_misses",PERF_TYPE_HW_CACHE,PERF_COUNT_HW_CACHE_LL|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16)}};
    for(auto& e:events) {
        perf_event_attr a{};a.size=sizeof(a);a.type=e.type;a.config=e.config;
        a.disabled=1;a.enable_on_exec=1;a.inherit=1;a.exclude_kernel=1;a.exclude_hv=1;
        a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
        e.fd=syscall(__NR_perf_event_open,&a,child,-1,-1,0);
        if(e.fd<0)std::cerr<<e.name<<": "<<std::strerror(errno)<<"\n";
    }
    kill(child,SIGCONT);waitpid(child,&status,0);
    std::cout<<"{\"perf_counters\":{";bool first=true;
    for(auto& e:events) if(e.fd>=0) {
        uint64_t v[3]{};ssize_t got=read(e.fd,v,sizeof(v));close(e.fd);
        if(got!=sizeof(v)){std::cerr<<"read failed "<<e.name<<"\n";continue;}
        if(!first)std::cout<<",";first=false;
        std::cout<<"\""<<e.name<<"\":{\"count\":"<<v[0]<<",\"enabled_ns\":"<<v[1]<<",\"running_ns\":"<<v[2]<<"}";
    }
    std::cout<<"}}\n";
    return WIFEXITED(status)?WEXITSTATUS(status):1;
}
