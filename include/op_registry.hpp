#pragma once
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct CorrectnessResult {
    bool ok;
    double metric;
    std::string metric_name;
    double threshold;
    std::string note;
};

struct PerfResult {
    double ms;             // GPU 执行时间（毫秒）
    double cpu_ms;         // CPU 执行时间（毫秒）
    std::string unit_name; // 性能单位（如 GB/s, TFLOP/s）
    double unit_value;     // 性能值（带宽或吞吐量）
    std::string note;      // 其他说明信息

    size_t input_size;                              // 输入数据大小（字节）
    size_t output_size;                             // 输出数据大小（字节）
    std::vector<std::vector<size_t>> input_format;  // 输入的形状
    std::vector<std::vector<size_t>> output_format; // 输出的形状
};

struct OpEntry {
    std::string name;
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
inline void register_op_t(const char *name, CorrectFn &&c, PerfFn &&p) {
    OpEntry e;
    e.name = name;
    e.correctness = std::function<CorrectnessResult()>(std::forward<CorrectFn>(c));
    e.performance = std::function<PerfResult()>(std::forward<PerfFn>(p));
    register_op_entry(std::move(e));
}

// 你现在用的：直接塞 lambda
#define REGISTER_OP(NAME, CORRECT_FN, PERF_FN)                                                                                                                 \
    namespace {                                                                                                                                                \
    struct _OpReg_##__LINE__ {                                                                                                                                 \
        _OpReg_##__LINE__() { register_op_t((NAME), (CORRECT_FN), (PERF_FN)); }                                                                                \
    };                                                                                                                                                         \
    static _OpReg_##__LINE__ _opreg_instance_##__LINE__;                                                                                                       \
    }

// 新增：塞“两个启动函数”（更清爽）
#define REGISTER_OP_FUNCS(NAME, CORRECT_FUNC, PERF_FUNC)                                                                                                       \
    namespace {                                                                                                                                                \
    struct _OpRegF_##__LINE__ {                                                                                                                                \
        _OpRegF_##__LINE__() { register_op_t((NAME), (CORRECT_FUNC), (PERF_FUNC)); }                                                                           \
    };                                                                                                                                                         \
    static _OpRegF_##__LINE__ _opregf_instance_##__LINE__;                                                                                                     \
    }
