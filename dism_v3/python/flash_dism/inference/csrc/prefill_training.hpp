#pragma once
// Native, bounded, reusable CPU batch planner. No Torch or CUDA dependency.
#include "prefill.hpp"
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <memory>
#include <chrono>

namespace dism_prefill {
class PlanningWorkers {
    std::mutex mutex, invocation;
    std::condition_variable cv;
    std::queue<std::function<void()>> queue;
    std::vector<std::thread> threads;
    bool stopping=false;
  public:
    explicit PlanningWorkers(int count) {
        if (count<1 || count>64) throw std::invalid_argument("workers must be in [1,64]");
        try {
            for (int i=0;i<count;++i) threads.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock,[this]{return stopping || !queue.empty();});
                        if (stopping && queue.empty()) return;
                        task=std::move(queue.front()); queue.pop();
                    }
                    task();
                }
            });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex); stopping=true; }
            cv.notify_all();
            for (auto &thread:threads) thread.join();
            throw;
        }
    }
    ~PlanningWorkers() {
        { std::lock_guard<std::mutex> lock(mutex); stopping=true; }
        cv.notify_all();
        for (auto &thread:threads) thread.join();
    }
    void run(size_t count,const std::function<void(size_t)> &fn) {
        // Serialize invocations, not jobs. Callers must separately bound the
        // number of outstanding plans to control retained metadata memory.
        std::lock_guard<std::mutex> invocation_lock(invocation);
        std::vector<std::future<void>> pending;
        pending.reserve(count);
        std::exception_ptr error;
        try {
            std::lock_guard<std::mutex> lock(mutex);
            for (size_t i=0;i<count;++i) {
                auto task=std::make_shared<std::packaged_task<void()>>([fn,i]{fn(i);});
                pending.push_back(task->get_future());
                queue.emplace([task]{(*task)();});
            }
        } catch (...) { error=std::current_exception(); }
        cv.notify_all();
        // Drain every job before propagating errors: jobs reference caller data.
        for (auto &future:pending) try { future.get(); } catch (...) { if (!error) error=std::current_exception(); }
        if (error) std::rethrow_exception(error);
    }
};

struct TrainingJob {
    int batch, head, start, end;
    double tau;
};

struct TrainingBuffers {
    int64_t *offsets;
    int *rows, *lengths, *resets, *stream_heads;
    float *prefixes, *weights, *dlog;
};

struct TrainingBatchProgram {
    std::shared_ptr<PlanningWorkers> workers;
    std::vector<TrainingJob> jobs;
    std::vector<ChunkProgram> chunks;
    std::vector<int64_t> stream_base{0}, chunk_base{0};
    int batch, heads, n, width, tau_rows;
    int64_t events=0;
    double plan_ms=0.;

    TrainingBatchProgram(std::shared_ptr<PlanningWorkers> pool,const int *labels,
                         int b,int h,int tokens,const double *tau,int tau_rows,
                         const int64_t *documents,int64_t docs,const uint8_t *hard,int c)
        : workers(std::move(pool)),batch(b),heads(h),n(tokens),width(c),tau_rows(tau_rows) {
        if (width!=16 && width!=32 && width!=64) throw std::invalid_argument("chunk width must be 16,32,64");
        if (int64_t(b)*h*tokens>std::numeric_limits<int>::max())
            throw std::invalid_argument("flattened vector rows exceed int32");
        std::vector<int64_t> covered(b,0);
        for (int64_t d=0;d<docs;++d) {
            int64_t row=documents[3*d],start=documents[3*d+1],end=documents[3*d+2];
            if (row<0 || row>=b || start!=covered[row] || end<=start || end>n)
                throw std::invalid_argument("documents must cover each row in order without gaps/overlap");
            covered[row]=end;
            for (int head=0;head<h;++head) if (hard[row*h+head])
                jobs.push_back({int(row),head,int(start),int(end),tau[(tau_rows==1 ? 0 : row)*h+head]});
        }
        for (auto end:covered) if (end!=n) throw std::invalid_argument("documents must cover every row completely");
        for (int64_t i=0;i<int64_t(b)*h;++i) if (hard[i]>1) throw std::invalid_argument("hard flags must be 0 or 1");
        for (int64_t i=0;i<int64_t(tau_rows)*h;++i) if (!std::isfinite(tau[i]))
            throw std::invalid_argument("finite tau required");
        chunks.resize(jobs.size());
        std::vector<int64_t> event_counts(jobs.size());
        auto begin=std::chrono::steady_clock::now();
        workers->run(jobs.size(),[&](size_t i) {
            const auto &job=jobs[i];
            int64_t base=(int64_t(job.batch)*h+job.head)*n+job.start;
            int count=job.end-job.start;
            Ids q(labels+base,labels+base+count);
            Ids k(labels+int64_t(b)*h*n+base,labels+int64_t(b)*h*n+base+count);
            auto p=Builder(q,k,Ids(count,0),job.tau,false,true).finish();
            event_counts[i]=p.rows.size();
            chunks[i]=chunk_program(p,width);
        });
        for (size_t i=0;i<jobs.size();++i) {
            stream_base.push_back(stream_base.back()+chunks[i].offsets.size()-1);
            chunk_base.push_back(chunk_base.back()+chunks[i].lengths.size());
            events+=event_counts[i];
        }
        plan_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }

    void pack_into(const TrainingBuffers &out) const {
        workers->run(jobs.size(),[&](size_t i) {
            const auto &p=chunks[i]; const auto &job=jobs[i];
            auto cb=chunk_base[i],sb=stream_base[i],slot=cb*width;
            int64_t token_base=int64_t(job.batch)*n+job.start;
            for (size_t e=0;e<p.rows.size();++e) {
                int row=p.rows[e];
                int64_t local=row>=0 ? row : -int64_t(row)-1;
                int absolute=int((token_base+local)*heads+job.head);
                out.rows[slot+e]=row>=0 ? absolute : -absolute-1;
            }
            std::copy(p.prefixes.begin(),p.prefixes.end(),out.prefixes+slot);
            std::copy(p.weights.begin(),p.weights.end(),out.weights+slot);
            std::copy(p.dlog.begin(),p.dlog.end(),out.dlog+slot);
            std::copy(p.lengths.begin(),p.lengths.end(),out.lengths+cb);
            std::copy(p.resets.begin(),p.resets.end(),out.resets+cb);
            // Exclude each job's terminal offset so adjacent jobs never race.
            for (size_t s=0;s+1<p.offsets.size();++s) {
                out.offsets[sb+s]=cb+p.offsets[s];
                out.stream_heads[sb+s]=(tau_rows==1 ? 0 : job.batch)*heads+job.head;
            }
        });
        out.offsets[stream_base.back()]=chunk_base.back();
    }
};
} // namespace dism_prefill
