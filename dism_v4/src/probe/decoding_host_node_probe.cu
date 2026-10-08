// Standalone proof: every graph replay runs GPU -> CPU -> GPU with fresh data.
// The callback performs pure CPU work only, with preallocated pinned buffers.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)

struct HostState { int *in, *out; int calls = 0; };
void CUDART_CB plan(void *data) {
    auto &s = *static_cast<HostState *>(data);
    *s.out = 3 * *s.in + 7;
    ++s.calls;
}
__global__ void produce(int *counter) { ++*counter; }
__global__ void consume(const int *input, int *output) { *output = *input + 1; }

int main() {
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int *counter, *input, *output;
    CHECK(cudaMalloc(&counter, sizeof(int)));
    CHECK(cudaMalloc(&input, sizeof(int)));
    CHECK(cudaMalloc(&output, sizeof(int)));
    CHECK(cudaMemset(counter, 0, sizeof(int)));
    HostState host;
    CHECK(cudaMallocHost(&host.in, sizeof(int)));
    CHECK(cudaMallocHost(&host.out, sizeof(int)));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    produce<<<1, 1, 0, stream>>>(counter);
    CHECK(cudaMemcpyAsync(host.in, counter, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaLaunchHostFunc(stream, plan, &host));
    CHECK(cudaMemcpyAsync(input, host.out, sizeof(int), cudaMemcpyHostToDevice, stream));
    consume<<<1, 1, 0, stream>>>(input, output);
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t executable;
    CHECK(cudaGraphInstantiate(&executable, graph, 0));
    std::vector<double> times;
    for (int i = 0; i < 320; ++i) {
        auto start = std::chrono::steady_clock::now();
        CHECK(cudaGraphLaunch(executable, stream));
        CHECK(cudaStreamSynchronize(stream));
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count();
        int result;
        CHECK(cudaMemcpy(&result, output, sizeof(int), cudaMemcpyDeviceToHost));
        if (result != 3 * (i+1) + 8 || host.calls != i+1) return 2;
        if (i >= 20) times.push_back(us);
    }
    std::sort(times.begin(), times.end());
    printf("{\"passed\":true,\"replays\":320,\"host_calls\":%d,\"median_us\":%.3f,\"p95_us\":%.3f}\n",
           host.calls, times[times.size()/2], times[times.size()*95/100]);
    CHECK(cudaGraphExecDestroy(executable));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(counter)); CHECK(cudaFree(input)); CHECK(cudaFree(output));
    CHECK(cudaFreeHost(host.in)); CHECK(cudaFreeHost(host.out));
}
