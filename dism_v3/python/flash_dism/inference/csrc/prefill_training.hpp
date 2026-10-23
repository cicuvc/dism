#pragma once
// Native, bounded, reusable CPU batch planner. No Torch or CUDA dependency.
#include "prefill.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <memory>
#include <chrono>

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

struct TrainingJob {
    int batch, head, start, end;
    double tau;
};

struct TrainingBuffers {
    int64_t *offsets;
    int *rows, *lengths, *resets, *stream_heads;
    float *prefixes, *weights, *dlog;
};

// Bucketed compact storage for single-chunk streams. Each bucket is itself a
// padded plan of one fixed width, so a consumer keeps its chunk*width indexing
// and only needs the block index instead of a global chunk index. Multi-chunk
// streams keep the ordinary C-wide padded layout (their chunks are ~full).
constexpr int compact_widths[3] = {16, 32, 64};
inline int compact_bucket(int length) {
    int bucket = 0;
    while (bucket < 2 && length > compact_widths[bucket]) ++bucket;
    return bucket;
}

struct CompactBucketBuffers {
    int *rows = nullptr, *lengths = nullptr, *heads = nullptr;
    float *prefixes = nullptr, *weights = nullptr, *dlog = nullptr;
};

struct TrainingBatchProgram {
    std::shared_ptr<PlanningWorkers> workers;
    std::vector<TrainingJob> jobs;
    // The padded ChunkProgram layout is materialized directly into the caller's
    // pinned buffer by pack_into(); only the compact per-job event program is
    // retained between the two phases. The final bytes are unchanged.
    std::vector<Program> programs;
    // Boundary discovery is shared by sizing and packing; never rescan events.
    struct Boundaries {
        std::vector<int64_t> ends, stream_chunks{0};
    };
    std::vector<Boundaries> boundaries;
    std::vector<int64_t> stream_base{0}, chunk_base{0};
    // Compact mode: single-chunk streams live in per-width buckets, the arrays
    // above then describe only the multi-chunk streams.
    bool compact = false;
    std::array<int64_t,3> bucket_blocks{0,0,0};
    std::vector<std::array<int64_t,3>> job_block_base;
    std::vector<int64_t> job_multi_streams;
    int batch, heads, n, width, tau_rows;
    int64_t events=0;
    double plan_ms=0.;

    TrainingBatchProgram(std::shared_ptr<PlanningWorkers> pool,const int *labels,
                         int b,int h,int tokens,const double *tau,int tau_rows,
                         const int64_t *documents,int64_t docs,const uint8_t *hard,int c,
                         bool compact_mode=false, bool gradients=true, const int *query_reset=nullptr)
        : workers(std::move(pool)),compact(compact_mode),batch(b),heads(h),n(tokens),width(c),
          tau_rows(tau_rows) {
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
        programs.resize(jobs.size()); boundaries.resize(jobs.size());
        std::vector<int64_t> event_counts(jobs.size()), chunk_counts(jobs.size(),0);
        std::vector<std::array<int64_t,3>> single_counts(jobs.size());
        std::vector<int64_t> multi_stream_counts(jobs.size(),0);
        auto begin=std::chrono::steady_clock::now();
        workers->run(jobs.size(),[&](size_t i) {
            const auto &job=jobs[i];
            int64_t base=(int64_t(job.batch)*h+job.head)*n+job.start;
            int count=job.end-job.start;
            // Worker-local scratch is retained across jobs, bounded by the
            // largest document handled by this worker. Program output owns its
            // arrays independently and remains valid through pack completion.
            static thread_local Builder builder;
            builder.build(labels+base,labels+int64_t(b)*h*n+base,
                          query_reset ? query_reset+base : nullptr,count,job.tau,false,gradients);
            auto p=builder.finish();
            event_counts[i]=p.rows.size();
            // Discover chunk boundaries once; slot arrays are built by pack_into*.
            auto &bounds=boundaries[i];
            bounds.stream_chunks.reserve(p.offsets.size());
            bounds.ends.reserve(p.rows.size()/width+p.offsets.size());
            for (size_t s=1;s<p.offsets.size();++s) {
                int64_t start=p.offsets[s-1], end=p.offsets[s], chunks=0;
                while (start<end) {
                    int64_t stop=std::min(start+int64_t(width),end);
                    for (int64_t e=start+1;e<stop;++e)
                        if (p.rows[e]>=0 && p.decay[e]==0.) { stop=e; break; }
                    bounds.ends.push_back(stop);
                    start=stop; ++chunks;
                }
                bounds.stream_chunks.push_back(bounds.ends.size());
                if (compact && chunks==1)
                    single_counts[i][compact_bucket(int(p.offsets[s]-p.offsets[s-1]))]++;
                else {
                    chunk_counts[i]+=chunks;
                    if (compact) multi_stream_counts[i]++;
                }
            }
            programs[i]=std::move(p);
        });
        job_block_base.assign(jobs.size()+1,{0,0,0});
        for (size_t i=0;i<jobs.size();++i) {
            if (compact) {
                stream_base.push_back(stream_base.back()+multi_stream_counts[i]);
                for (int k=0;k<3;++k)
                    job_block_base[i+1][k]=job_block_base[i][k]+single_counts[i][k];
            } else
                stream_base.push_back(stream_base.back()+programs[i].offsets.size()-1);
            chunk_base.push_back(chunk_base.back()+chunk_counts[i]);
            events+=event_counts[i];
        }
        if (compact) bucket_blocks=job_block_base.back();
        job_multi_streams=multi_stream_counts;
        plan_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }

    void pack_into(const TrainingBuffers &out) const {
        workers->run(jobs.size(),[&](size_t i) {
            const auto &p=programs[i]; const auto &job=jobs[i];
            const auto &bounds=boundaries[i]; size_t boundary=0;
            int64_t cb=chunk_base[i], sb=stream_base[i];
            int64_t token_base=int64_t(job.batch)*n+job.start;
            const bool has_dlog=!p.dlogweight.empty();
            int head_offset=(tau_rows==1 ? 0 : job.batch)*heads+job.head;
            // Padding lanes reproduce the previous packed convention: the
            // relabeled absolute index of this job's local row 0. Consumers mask
            // by chunk length, but tests compare the whole buffer.
            int pad_row=int(token_base*heads+job.head);
            int64_t local_cb=cb;
            // Streamless jobs write nothing, matching the previous contract that
            // adjacent jobs never race on a shared boundary entry.
            if (p.offsets.size()>1) out.offsets[sb]=cb;
            for (size_t stream=1;stream<p.offsets.size();++stream) {
                int64_t start=p.offsets[stream-1], end=p.offsets[stream];
                while (start<end) {
                    int64_t stop=bounds.ends[boundary++];
                    int length=int(stop-start);
                    int reset=int(p.rows[start]>=0 && p.decay[start]==0.);
                    int *rows=out.rows+cb*width;
                    float *prefixes=out.prefixes+cb*width, *weights=out.weights+cb*width;
                    float *dlog=has_dlog ? out.dlog+cb*width : nullptr;
                    double prefix=0.;
                    for (int k=0;k<length;++k) {
                        int64_t e=start+k;
                        if (p.rows[e]>=0 && !(reset && k==0)) prefix+=std::log(p.decay[e]);
                        int row=p.rows[e];
                        int64_t local=row>=0 ? row : -int64_t(row)-1;
                        int absolute=int((token_base+local)*heads+job.head);
                        rows[k]=row>=0 ? absolute : -absolute-1;
                        prefixes[k]=float(prefix);
                        weights[k]=float(p.weight[e]);
                        if (dlog) dlog[k]=float(p.dlogweight[e]);
                    }
                    for (int k=length;k<width;++k) {
                        rows[k]=pad_row; prefixes[k]=0.f; weights[k]=0.f;
                        if (dlog) dlog[k]=0.f;
                    }
                    out.lengths[cb]=length;
                    out.resets[cb]=reset;
                    start=stop; ++cb; ++local_cb;
                }
                // p.offsets holds event offsets; the packed layout uses chunk
                // counts, so stream boundaries come from the running walk.
                if (stream+1<p.offsets.size()) out.offsets[sb+stream]=local_cb;
                if (out.stream_heads) out.stream_heads[sb+stream-1]=head_offset;
            }
        },true);
        out.offsets[stream_base.back()]=chunk_base.back();
    }

    // Compact mode: single-chunk streams go to per-width bucket blocks, all
    // other streams keep the ordinary C-wide padded layout. Event values, chunk
    // boundaries and the padding convention are shared with pack_into().
    void pack_into_compact(const std::vector<CompactBucketBuffers> &bk,
                           const TrainingBuffers &multi) const {
        workers->run(jobs.size(),[&](size_t i) {
            const Program &p=programs[i]; const auto &job=jobs[i];
            const auto &bounds=boundaries[i]; size_t boundary=0;
            std::array<int64_t,3> blk=job_block_base[i];
            int64_t mc=chunk_base[i], sb=stream_base[i];
            const int64_t token_base=int64_t(job.batch)*n+job.start;
            const int head_offset=(tau_rows==1 ? 0 : job.batch)*heads+job.head;
            const int pad_row=int(token_base*heads+job.head);
            const bool has_dlog=!p.dlogweight.empty();
            // Multi-stream ordinal within this job; jobs without multi-chunk
            // streams write nothing so adjacent jobs never race on a boundary.
            const int64_t job_ms=job_multi_streams[i];
            int64_t msl=0;
            for (size_t s=1;s<p.offsets.size();++s) {
                const int64_t stream_start=p.offsets[s-1], stream_end=p.offsets[s];
                const int64_t chunks=bounds.stream_chunks[s]-bounds.stream_chunks[s-1];
                if (chunks==1) {
                    ++boundary;
                    const int length=int(stream_end-stream_start);
                    const int b=compact_bucket(length), w=compact_widths[b];
                    const int64_t block=blk[b]++;
                    int *rows=bk[b].rows+block*w;
                    float *prefixes=bk[b].prefixes+block*w, *weights=bk[b].weights+block*w;
                    float *dlog=has_dlog ? bk[b].dlog+block*w : nullptr;
                    const int reset=int(p.rows[stream_start]>=0 && p.decay[stream_start]==0.);
                    double prefix=0.;
                    for (int k=0;k<length;++k) {
                        const int64_t e=stream_start+k;
                        if (p.rows[e]>=0 && !(reset && k==0)) prefix+=std::log(p.decay[e]);
                        const int row=p.rows[e];
                        const int64_t local=row>=0 ? row : -int64_t(row)-1;
                        const int absolute=int((token_base+local)*heads+job.head);
                        rows[k]=row>=0 ? absolute : -absolute-1;
                        prefixes[k]=float(prefix); weights[k]=float(p.weight[e]);
                        if (dlog) dlog[k]=float(p.dlogweight[e]);
                    }
                    for (int k=length;k<w;++k) {
                        rows[k]=pad_row; prefixes[k]=0.f; weights[k]=0.f;
                        if (dlog) dlog[k]=0.f;
                    }
                    bk[b].lengths[block]=length; bk[b].heads[block]=head_offset;
                    continue;
                }
                multi.offsets[sb+msl]=mc;
                int64_t cursor2=stream_start;
                while (cursor2<stream_end) {
                    int64_t stop=bounds.ends[boundary++];
                    const int length=int(stop-cursor2);
                    const int reset=int(p.rows[cursor2]>=0 && p.decay[cursor2]==0.);
                    int *rows=multi.rows+mc*width;
                    float *prefixes=multi.prefixes+mc*width, *weights=multi.weights+mc*width;
                    float *dlog=has_dlog ? multi.dlog+mc*width : nullptr;
                    double prefix=0.;
                    for (int k=0;k<length;++k) {
                        const int64_t e=cursor2+k;
                        if (p.rows[e]>=0 && !(reset && k==0)) prefix+=std::log(p.decay[e]);
                        const int row=p.rows[e];
                        const int64_t local=row>=0 ? row : -int64_t(row)-1;
                        const int absolute=int((token_base+local)*heads+job.head);
                        rows[k]=row>=0 ? absolute : -absolute-1;
                        prefixes[k]=float(prefix); weights[k]=float(p.weight[e]);
                        if (dlog) dlog[k]=float(p.dlogweight[e]);
                    }
                    for (int k=length;k<width;++k) {
                        rows[k]=pad_row; prefixes[k]=0.f; weights[k]=0.f;
                        if (dlog) dlog[k]=0.f;
                    }
                    multi.lengths[mc]=length; multi.resets[mc]=reset;
                    cursor2=stop; ++mc;
                }
                multi.stream_heads[sb+msl]=head_offset;
                ++msl;
            }
            if (msl!=job_ms) throw std::logic_error("multi-chunk stream count mismatch");
        },true);
        multi.offsets[stream_base.back()]=chunk_base.back();
    }
};
} // namespace dism_prefill
