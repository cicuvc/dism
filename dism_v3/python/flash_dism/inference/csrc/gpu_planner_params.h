#pragma once
// Device-only POD; no Torch headers in the CUDA translation unit.
struct GpuPlannerArgs {
    void *keys, *values;
    const float *summaries, *counts, *tau;
    const int *topology, *edges;
    int *state, *band, *history;
    int *marks, *slots, *tasks;
    float *coefficients, *output;
    int heads, r, d, capacity, interval, snapshot, nodes, edge_capacity, task_capacity;
};
