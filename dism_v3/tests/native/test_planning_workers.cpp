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
    // A running invocation must not prevent an independent caller submitting
    // work to an idle worker. Also exercise cross-invocation exception drain.
    dism_prefill::PlanningWorkers concurrent(2);
    std::atomic<bool> entered{false}, released{false};
    std::thread first([&] {
        concurrent.run(1,[&](size_t) {
            entered=true;
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while (!released && std::chrono::steady_clock::now()<deadline)
                std::this_thread::yield();
            assert(released);
        });
    });
    while (!entered) std::this_thread::yield();
    concurrent.run(1,[&](size_t){released=true;},true);
    first.join();
    auto failing=[&] {
        for (int i=0;i<500;++i) {
            std::atomic<int> completed{0}; bool error=false;
            try { concurrent.run(11,[&](size_t j) {
                ++completed;
                if (j==2) throw std::runtime_error("concurrent expected");
            },i%2); } catch(const std::runtime_error&) { error=true; }
            assert(error && completed==11);
        }
    };
    std::thread c(failing),d(failing);c.join();d.join();
}
