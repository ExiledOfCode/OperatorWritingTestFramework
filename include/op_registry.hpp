#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct CorrectnessResult {
    bool ok = false;
    double metric = 0.0;
    std::string metric_name;
    double threshold = 0.0;
    std::string note;
};

struct MetricEntry {
    std::string name;
    double value = 0.0;
    std::string unit;
};

struct PerfResult {
    double ms = 0.0;
    double cpu_ms = 0.0;
    std::string unit_name;
    double unit_value = 0.0;
    std::string note;

    size_t input_size = 0;
    size_t output_size = 0;
    std::vector<std::vector<size_t>> input_format;
    std::vector<std::vector<size_t>> output_format;
    std::vector<MetricEntry> extra_metrics;
};

struct ParamSpec {
    std::string name;
    size_t default_value = 0;
};

struct OpMetadata {
    std::string name;
    std::string kind;
    std::vector<ParamSpec> params;
    std::vector<std::string> edge_axes;
    std::vector<std::string> baselines;
    std::string description;
};

struct OpEntry {
    std::string name;
    OpMetadata meta;
    std::function<CorrectnessResult()> correctness;
    std::function<PerfResult()> performance;
};

inline std::vector<OpEntry> &op_registry() {
    static std::vector<OpEntry> reg;
    return reg;
}

inline void register_op_entry(OpEntry e) {
    static std::mutex m;
    std::lock_guard<std::mutex> g(m);
    op_registry().push_back(std::move(e));
}

template <class CorrectFn, class PerfFn>
inline void register_op_t(OpMetadata meta, CorrectFn &&correct_fn, PerfFn &&perf_fn) {
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = std::move(meta);
    entry.correctness = std::function<CorrectnessResult()>(std::forward<CorrectFn>(correct_fn));
    entry.performance = std::function<PerfResult()>(std::forward<PerfFn>(perf_fn));
    register_op_entry(std::move(entry));
}

#define OPFW_CONCAT_INNER(a, b) a##b
#define OPFW_CONCAT(a, b) OPFW_CONCAT_INNER(a, b)

#define OPFW_REGISTER_ENTRY(ENTRY_EXPR)                                                                                                                         \
    namespace {                                                                                                                                                \
    struct OPFW_CONCAT(_OpReg_, __LINE__) {                                                                                                                   \
        OPFW_CONCAT(_OpReg_, __LINE__)() { register_op_entry((ENTRY_EXPR)); }                                                                                 \
    };                                                                                                                                                         \
    static OPFW_CONCAT(_OpReg_, __LINE__) OPFW_CONCAT(_opreg_instance_, __LINE__);                                                                           \
    }
