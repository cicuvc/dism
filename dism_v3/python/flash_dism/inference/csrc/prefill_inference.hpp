#pragma once
// Inference uses the same bounded pool, scratch and direct packing as training,
// with gradients disabled. No Torch/CUDA or Python objects in worker jobs.
#include "prefill_training.hpp"

namespace dism_prefill {
struct PrefillRoutingBuffers {
    int64_t *stream_ids, *chunk_ids, *local_offsets, *short_stream_ids;
    int64_t *long_stream_ids, *long_chunk_ids, *long_local_offsets;
};

struct PrefillBatchProgram {
    std::shared_ptr<TrainingBatchProgram> program;
    std::vector<int64_t> short_base{0}, long_base{0}, long_chunk_base{0};
    int threshold;

    PrefillBatchProgram(std::shared_ptr<PlanningWorkers> pool,const int *labels,
                       int b,int h,int n,const double *tau,int tau_rows,
                       const int *reset,int width,int dispatch_threshold)
        : threshold(dispatch_threshold) {
        std::vector<int64_t> docs;
        for (int row=0;row<b;++row) { docs.push_back(row); docs.push_back(0); docs.push_back(n); }
        std::vector<uint8_t> hard(size_t(b)*h,1);
        program=std::make_shared<TrainingBatchProgram>(std::move(pool),labels,b,h,n,tau,tau_rows,
            docs.data(),b,hard.data(),width,false,false,reset);
        // Only prefix counts are retained. Dispatch arrays are written directly
        // to the final pinned allocation alongside the packed event arrays.
        for (const auto &bounds:program->boundaries) {
            int64_t ns=0,nl=0,nc=0;
            for (size_t s=1;s<bounds.stream_chunks.size();++s) {
                int64_t chunks=bounds.stream_chunks[s]-bounds.stream_chunks[s-1];
                if (chunks>threshold) { ++nl; nc+=chunks; } else ++ns;
            }
            short_base.push_back(short_base.back()+ns);
            long_base.push_back(long_base.back()+nl);
            long_chunk_base.push_back(long_chunk_base.back()+nc);
        }
    }

    void pack_into(const TrainingBuffers &out,const PrefillRoutingBuffers &ids) const {
        program->pack_into(out);
        program->workers->run(program->jobs.size(),[&](size_t i) {
            const auto &bounds=program->boundaries[i];
            int64_t sb=program->stream_base[i], cb=program->chunk_base[i];
            int64_t ns=short_base[i], nl=long_base[i], nc=long_chunk_base[i];
            for (size_t s=1;s<bounds.stream_chunks.size();++s) {
                int64_t stream=sb+int64_t(s)-1;
                int64_t begin=cb+bounds.stream_chunks[s-1], end=cb+bounds.stream_chunks[s];
                ids.stream_ids[stream]=stream;
                ids.local_offsets[stream]=begin;
                if (end-begin>threshold) {
                    ids.long_stream_ids[nl]=stream;
                    ids.long_local_offsets[nl++]=nc;
                    for (int64_t c=begin;c<end;++c) ids.long_chunk_ids[nc++]=c;
                } else ids.short_stream_ids[ns++]=stream;
            }
            for (int64_t c=cb;c<program->chunk_base[i+1];++c) ids.chunk_ids[c]=c;
        },true);
    }
};
} // namespace dism_prefill
