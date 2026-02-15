#pragma once
#include <functional>
#include <string>
#include <vector>
#include <mutex>
#include <utility>

struct CorrectnessResult {
  bool ok;
  double metric;
  std::string metric_name;
  double threshold;
  std::string note;
};

struct PerfResult {
  double ms;
  std::string unit_name;
  double unit_value;
  std::string note;
};

struct OpEntry {
  std::string name;
  std::function<CorrectnessResult()> correctness;
  std::function<PerfResult()> performance;
};

inline std::vector<OpEntry>& op_registry() {
  static std::vector<OpEntry> reg;
  return reg;
}

inline void register_op_entry(OpEntry e) {
  static std::mutex m;
  std::lock_guard<std::mutex> g(m);
  op_registry().push_back(std::move(e));
}

template <class CorrectFn, class PerfFn>
inline void register_op_t(const char* name, CorrectFn&& c, PerfFn&& p) {
  OpEntry e;
  e.name = name;
  e.correctness = std::function<CorrectnessResult()>(std::forward<CorrectFn>(c));
  e.performance = std::function<PerfResult()>(std::forward<PerfFn>(p));
  register_op_entry(std::move(e));
}

// 你现在用的：直接塞 lambda
#define REGISTER_OP(NAME, CORRECT_FN, PERF_FN)                    \
  namespace {                                                     \
  struct _OpReg_##__LINE__ {                                      \
    _OpReg_##__LINE__() {                                         \
      register_op_t((NAME), (CORRECT_FN), (PERF_FN));             \
    }                                                             \
  };                                                              \
  static _OpReg_##__LINE__ _opreg_instance_##__LINE__;            \
  }

// 新增：塞“两个启动函数”（更清爽）
#define REGISTER_OP_FUNCS(NAME, CORRECT_FUNC, PERF_FUNC)          \
  namespace {                                                     \
  struct _OpRegF_##__LINE__ {                                     \
    _OpRegF_##__LINE__() {                                        \
      register_op_t((NAME), (CORRECT_FUNC), (PERF_FUNC));         \
    }                                                             \
  };                                                              \
  static _OpRegF_##__LINE__ _opregf_instance_##__LINE__;          \
  }
