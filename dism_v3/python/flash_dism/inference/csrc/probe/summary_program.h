#pragma once
#include <cuda_runtime.h>
struct SummaryProgramArgs {
    const void *keys, *values;
    const int *heads, *nodes, *chain_offsets, *chain_nodes, *ancestor_offsets, *ancestors;
    const int *prefix_map, *job_ids;
    int *job_slots;
    const float *weights, *rho, *chain_scale, *ancestor_scale;
    float *prefix, *block_carry, *chain_tail, *output;
    int *queue, *ready, *control;
    int head_count, capacity, r, d, tile, tiles, block_size, queue_capacity, initial_tasks, jobs;
    int slots, prefix_pitch, carry_pitch;
};
extern "C" void launch_summary_program(SummaryProgramArgs, const int *, bool, int, cudaStream_t);
