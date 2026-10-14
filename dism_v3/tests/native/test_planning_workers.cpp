// Build with -pthread and optionally -fsanitize=thread or address,undefined.
#include "prefill_training.hpp"
#include <cassert>
#include <stdexcept>

int main() {
    dism_prefill::PlanningWorkers pool(4);
    // Exposes early destruction/reuse of the stack-owned Done on tiny jobs.
    for (int i=0; i<100000; ++i) pool.run(1, [](size_t){});
    std::atomic<int> calls{0};
    bool caught=false;
    try {
        pool.run(64, [&](size_t i) {
            ++calls;
            if (i==5) throw std::runtime_error("expected");
        });
    } catch (const std::runtime_error&) { caught=true; }
    assert(caught && calls==64);
    pool.run(0, [](size_t){assert(false);});
    auto submit = [&] {
        for (int i=0; i<1000; ++i) pool.run(7, [&](size_t){++calls;});
    };
    std::thread a(submit), b(submit);
    a.join(); b.join();
    assert(calls==14064);
}
