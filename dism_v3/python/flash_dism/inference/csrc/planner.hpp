#pragma once

// CPU control plane. All vector payloads remain outside this class (on GPU).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dism_decode {

struct Node {
    int link = -1, length = 0, position = -1;
    std::unordered_map<int, int> next;
};

struct Task {
    // 0: raw position, 1: materialized subtree, 2: sampled prefix.
    int kind, index;
    double coefficient;
};

// sum_{i=0}^{length-1} exp(-tau*i), stable including tau == 0.
inline double geometric(int length, double tau) {
    if (length <= 0)
        return 0.;
    if (tau == 0.)
        return double(length);
    return -std::expm1(-tau * length) / -std::expm1(-tau);
}

class Planner {
  public:
    const int interval, sample_interval, threshold, rank;
    const double tau;
    std::vector<int> keys, queries, resets;
    std::vector<Node> nodes;
    std::vector<std::vector<int>> children;
    std::vector<int> order, materialized, samples, nearest, mat_index, sample_index;
    std::vector<int> band;
    int snapshot_size = 0, state = 0, match = 0;
    int max_length = 0;
    double fallback = 1.;
    double inv_denominator = 1.;
    std::vector<double> subtree_count, prefix_weight;
    std::vector<double> edge_weight, edge_decay;
    int cached_state = -1, cached_match = -1;
    std::vector<Task> cached_snapshot_tasks;

    Planner(int r, int b, int k, int t, double tau_)
        : interval(b), sample_interval(k), threshold(t), rank(r), tau(tau_) {
        if (r <= 0 || b <= 0 || k <= 0 || t <= r || tau < 0 || !std::isfinite(tau))
            throw std::invalid_argument("require r,b,k>0, threshold>r, finite tau>=0");
        rebuild();
    }

    bool needs_rebuild() const { return int(keys.size()) - snapshot_size >= interval; }

    // Initialize a prefill history without evaluating all intermediate outputs.
    // The last band entries may exceed interval: walk complete diagonals, not
    // just the most recent interval query tokens. Reset applies before its row.
    void initialize(const std::vector<int> &s, const std::vector<int> &q, const std::vector<int> &reset) {
        if (!keys.empty() || s.size() != q.size() || s.size() != reset.size())
            throw std::invalid_argument("initialize requires empty cache and equal history lengths");
        keys = s;
        queries = q;
        resets = reset;
        int n = int(s.size()), start = std::max(0, n - interval);
        band.assign(n - start, 0);
        for (int p = start; p < n; ++p) {
            int length = 0;
            while (p - length >= 0 && n - 1 - length >= 0 && s[p - length] == q[n - 1 - length]) {
                ++length;
                if (reset[n - length])
                    break;
            }
            band[p - start] = length;
        }
    }

    void extend(int character, int position, int &last) {
        int current = int(nodes.size());
        nodes.push_back(Node{});
        nodes[current].length = nodes[last].length + 1;
        nodes[current].position = position;
        int p = last;
        while (p >= 0 && !nodes[p].next.count(character)) {
            nodes[p].next[character] = current;
            p = nodes[p].link;
        }
        if (p < 0)
            nodes[current].link = 0;
        else {
            int q = nodes[p].next.at(character);
            if (nodes[p].length + 1 == nodes[q].length)
                nodes[current].link = q;
            else {
                int clone = int(nodes.size());
                Node copied = nodes[q];
                copied.length = nodes[p].length + 1;
                copied.position = -1;
                nodes.push_back(std::move(copied));
                while (p >= 0) {
                    auto it = nodes[p].next.find(character);
                    if (it == nodes[p].next.end() || it->second != q)
                        break;
                    it->second = clone;
                    p = nodes[p].link;
                }
                nodes[current].link = nodes[q].link = clone;
            }
        }
        last = current;
    }

    void advance(int character, bool reset) {
        if (reset) {
            state = 0;
            match = 0;
        }
        while (state && !nodes[state].next.count(character)) {
            state = nodes[state].link;
            match = std::min(match, nodes[state].length);
        }
        auto it = nodes[state].next.find(character);
        if (it == nodes[state].next.end()) {
            state = 0;
            match = 0;
        } else {
            state = it->second;
            ++match;
        }
    }

    // Call before append when needs_rebuild(). Caller builds matching GPU matrices
    // before consuming subsequent plans. Band is deliberately NOT cleared.
    void rebuild() {
        cached_state = cached_match = -1;
        cached_snapshot_tasks.clear();
        nodes.clear();
        nodes.push_back(Node{});
        int last = 0;
        for (int p = 0; p < int(keys.size()); ++p)
            extend(keys[p], p, last);
        snapshot_size = int(keys.size());
        children.assign(nodes.size(), {});
        for (int n = 1; n < int(nodes.size()); ++n)
            children[nodes[n].link].push_back(n);
        order.clear();
        std::vector<int> stack{0};
        while (!stack.empty()) {
            int n = stack.back();
            stack.pop_back();
            order.push_back(n);
            for (auto it = children[n].rbegin(); it != children[n].rend(); ++it)
                stack.push_back(*it);
        }
        materialized.clear();
        samples.clear();
        mat_index.assign(nodes.size(), -1);
        sample_index.assign(nodes.size(), -1);
        std::vector<int> cost(nodes.size(), 1), height(nodes.size(), 0);
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            int n = *it, c = 1, h = 0;
            for (int child : children[n]) {
                c += cost[child];
                if (height[child] >= 0)
                    h = std::max(h, height[child] + 1);
            }
            // The root represents length zero and is never a queried range.
            // Materializing it wastes an RD matrix and can trigger an otherwise
            // completely unnecessary GPU rebuild on high-entropy streams.
            if (n && c > threshold) {
                mat_index[n] = int(materialized.size());
                materialized.push_back(n);
                cost[n] = rank;
            } else
                cost[n] = c;
            if (n && h == sample_interval - 1) {
                sample_index[n] = int(samples.size());
                samples.push_back(n);
                height[n] = -1;
            } else
                height[n] = h;
        }
        nearest.assign(nodes.size(), 0);
        for (int n : order)
            if (n)
                nearest[n] = sample_index[n] >= 0 ? n : nearest[nodes[n].link];
        // Denominator has no readout dependence. Build scalar summaries once on
        // CPU in FP64, alongside the topology, rather than on every GPU channel.
        subtree_count.assign(nodes.size(), 0.);
        prefix_weight.assign(nodes.size(), 0.);
        edge_weight.assign(nodes.size(), 0.);
        edge_decay.assign(nodes.size(), 0.);
        for (int n : order)
            subtree_count[n] = nodes[n].position >= 0 ? 1. : 0.;
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            if (*it)
                subtree_count[nodes[*it].link] += subtree_count[*it];
        for (int n : order)
            if (n) {
                int parent = nodes[n].link, gap = nodes[n].length - nodes[parent].length;
                edge_weight[n] = geometric(gap, tau);
                edge_decay[n] = std::exp(-tau * gap);
                prefix_weight[n] = edge_decay[n] * prefix_weight[parent] + edge_weight[n] * subtree_count[n];
            }
        state = 0;
        match = 0;
        for (int p = 0; p < int(queries.size()); ++p)
            advance(queries[p], resets[p]);
    }

    void add_range(int root, double coefficient, std::vector<Task> &tasks) const {
        if (coefficient == 0.)
            return;
        std::vector<int> stack{root};
        while (!stack.empty()) {
            int n = stack.back();
            stack.pop_back();
            if (mat_index[n] >= 0)
                tasks.push_back({1, mat_index[n], coefficient});
            else {
                if (nodes[n].position >= 0)
                    tasks.push_back({0, nodes[n].position, coefficient});
                for (int child : children[n])
                    stack.push_back(child);
            }
        }
    }

    std::vector<Task> append(int key, int query, bool reset = false) {
        if (needs_rebuild())
            throw std::logic_error("rebuild required before append");
        int old_n = int(keys.size()), old_start = old_n - int(band.size());
        keys.push_back(key);
        queries.push_back(query);
        resets.push_back(reset);
        int start = std::max(0, int(keys.size()) - interval);
        std::vector<int> next_band(int(keys.size()) - start, 0);
        for (int p = start; p < int(keys.size()); ++p)
            if (keys[p] == query) {
                int previous = (!reset && p > 0) ? band.at(p - 1 - old_start) : 0;
                next_band[p - start] = previous + 1;
            }
        band.swap(next_band);
        advance(query, reset);
        max_length = match;
        for (int p = snapshot_size; p < int(keys.size()); ++p)
            max_length = std::max(max_length, band.at(p - start));
        fallback = std::exp(-tau * max_length);
        // A frozen matcher often stays at the same state/length (particularly
        // repeated symbols). Cache its already-coalesced vector work, not an
        // unbounded map of historical query plans. Tail scaling remains fresh.
        if (cached_state != state || cached_match != match) {
            std::vector<Task> snapshot_tasks;
            if (match) {
                int parent = nodes[state].link, sample = nearest[parent];
                if (sample)
                    snapshot_tasks.push_back({2, sample_index[sample], std::exp(tau * (nodes[sample].length - match))});
                for (int n = parent; n != sample; n = nodes[n].link)
                    add_range(n,
                              std::exp(tau * (nodes[n].length - match)) *
                                  geometric(nodes[n].length - nodes[nodes[n].link].length, tau),
                              snapshot_tasks);
                add_range(state, geometric(match - nodes[parent].length, tau), snapshot_tasks);
            }
            cached_snapshot_tasks = coalesce(snapshot_tasks);
            cached_state = state;
            cached_match = match;
        }
        std::vector<Task> tasks = cached_snapshot_tasks;
        double scale = std::exp(tau * (match - max_length));
        for (auto &task : tasks)
            task.coefficient *= scale;
        for (int p = snapshot_size; p < int(keys.size()); ++p) {
            int length = band.at(p - start);
            if (length)
                tasks.push_back({0, p, std::exp(tau * (length - max_length)) * geometric(length, tau)});
        }
        double denominator = fallback;
        for (auto task : tasks) {
            double count = task.kind == 0   ? 1.
                           : task.kind == 1 ? subtree_count[materialized[task.index]]
                                            : prefix_weight[samples[task.index]];
            denominator += task.coefficient * count;
        }
        inv_denominator = 1. / denominator;
        // Snapshot and tail raw positions are disjoint; no second sort needed.
        return tasks;
    }

    static std::vector<Task> coalesce(std::vector<Task> &tasks) {
        std::sort(tasks.begin(), tasks.end(),
                  [](const Task &a, const Task &b) { return std::pair(a.kind, a.index) < std::pair(b.kind, b.index); });
        std::vector<Task> merged;
        for (auto task : tasks) {
            if (!merged.empty() && merged.back().kind == task.kind && merged.back().index == task.index)
                merged.back().coefficient += task.coefficient;
            else
                merged.push_back(task);
        }
        return merged;
    }
};
} // namespace dism_decode
