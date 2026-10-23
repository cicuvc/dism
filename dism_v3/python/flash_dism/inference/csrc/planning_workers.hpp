#pragma once
// Shared bounded CPU worker pool. No Torch or CUDA dependency.
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dism_prefill {
class PlanningWorkers {
    // One stack-owned submission per run, rather than one queue entry per
    // job. Multiple callers may submit concurrently; workers are still bounded.
    struct Batch {
        const std::function<void(size_t)> &fn;
        size_t count, next=0, remaining;
        bool pack;
        std::condition_variable cv;
        std::exception_ptr error;
        Batch(const std::function<void(size_t)> &fn_,size_t n,bool pack_)
            : fn(fn_),count(n),remaining(n),pack(pack_) {}
    };
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Batch*> queue;
    std::vector<std::thread> threads;
    bool stopping=false;
  public:
    explicit PlanningWorkers(int count) {
        if (count<1 || count>64) throw std::invalid_argument("workers must be in [1,64]");
        try {
            for (int i=0;i<count;++i) threads.emplace_back([this] {
                for (;;) {
                    Batch *batch; size_t index;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock,[this]{return stopping || !queue.empty();});
                        if (stopping && queue.empty()) return;
                        // Packing releases retained programs and lets the caller
                        // upload immediately. FIFO within each stage prevents
                        // arbitrary overtaking between equal-priority callers.
                        auto it=std::find_if(queue.begin(),queue.end(),[](Batch *b){return b->pack;});
                        if (it==queue.end()) it=queue.begin();
                        batch=*it; index=batch->next++;
                        if (batch->next==batch->count) queue.erase(it);
                    }
                    std::exception_ptr error;
                    try { batch->fn(index); } catch (...) { error=std::current_exception(); }
                    {
                        // Same mutex protects dispatch, completion, and waiter
                        // lifetime. No worker may touch Batch after notifying.
                        std::lock_guard<std::mutex> lock(mutex);
                        if (error && !batch->error) batch->error=error;
                        if (--batch->remaining==0) batch->cv.notify_one();
                    }
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
    void run(size_t count,const std::function<void(size_t)> &fn,bool pack=false) {
        if (!count) return;
        Batch batch(fn,count,pack);
        std::unique_lock<std::mutex> lock(mutex);
        // A failed push exposes no jobs; every successful submission is drained
        // before run returns, including exceptions from individual jobs.
        queue.push_back(&batch);
        cv.notify_all();
        batch.cv.wait(lock,[&]{return batch.remaining==0;});
        if (batch.error) std::rethrow_exception(batch.error);
    }
};

} // namespace dism_prefill
