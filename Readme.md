# CUDA 算子测试框架

现在的框架是三层：

1. `src/**/*.cu`
   每个注册了 `LC_REGISTER_*` 的源文件都会被编译成一个独立可执行文件。
2. `tests/op_runner.cpp`
   负责把单个算子包装成统一 CLI，支持 `--describe`、`--mode correctness|performance|both` 和 JSON 输出。
3. `scripts/run_benchmarks.py`
   负责读取测试配置、按算子名调度可执行文件、执行 baseline、产出 CSV / 文本报告 / SVG 折线图。
4. `scripts/select_build_targets.py`
   负责从 YAML 里提取 op 名，映射到需要构建的 CMake target。

## 运行方式

默认入口：

```bash
./run.sh
```

指定 YAML 配置：

```bash
./run.sh configs/benchmark_plan.yaml
```

现在默认只构建当前配置里出现的 op，以及显式写在 `ops[].baselines` 里的 baseline。

如果你想强制全量构建：

```bash
BUILD_TARGET_MODE=all ./run.sh
```

只构建不跑：

```bash
./debug_build.sh
./ncu_build.sh
```

单独跑一个 op：

```bash
python3 scripts/run_op.py \
  --build-dir build_framework_check \
  --op gemm_demo1 \
  --mode correctness \
  --param M=256 --param N=256 --param K=256
```

## 配置格式

配置文件现在用 YAML。

原因：

- 支持注释
- IDE 高亮普遍更好
- 手改比 JSON 更轻松

示例见 [benchmark_plan.yaml](/mnt/d/Project_Repository/Lab_Projects/CUDA_Operator_Library/OperatorWritingTestFramework/configs/benchmark_plan.yaml)。

核心字段：

- `global.report_dir`
  报告输出目录
- `global.warmup / global.iters / global.seed`
  benchmark 默认参数
- `global.check_correctness`
  是否做正确性检查
- `ops[].name`
  要测试的算子名
- `ops[].baselines`
  baseline 列表，例如 `cublas_gemm_f32`。现在只以 YAML 里显式写出来的为准。
- `ops[].params`
  固定参数，例如 `K: 16`
- `ops[].edge`
  区间扫描，格式是 `from / to / stride`
- `ops[].shapes`
  显式给出若干 shape

说明：

- `edge` 用于扫这个 op 暴露出来的 edge 轴。
- `shapes` 用于你想精确控制某几个形状的时候。
- 一个 op 可以只写 `params`，那就只跑默认 shape 上的单个 case。
- 构建阶段也会按 `ops[].name + ops[].baselines` 只编译需要的 target。

## 输出结果

运行后会在 `reports/...` 下生成：

- `benchmark_measurements.csv`
  每次测量的原始结果
- `benchmark_comparisons.csv`
  主算子和 baseline 的速度对比
- `benchmark_summary.txt`
  文字摘要
- `plots/*.svg`
  不同数据量下的延迟 / 吞吐折线图
- `raw/*.json`
  每个 case 的原始 runner 输出

## 算子接入

注册和公共 helper 在 [op_runtime.hpp](/mnt/d/Project_Repository/Lab_Projects/CUDA_Operator_Library/OperatorWritingTestFramework/include/op_runtime.hpp)。

常用注册宏：

- `LC_REGISTER_GEMM_OP`
- `LC_REGISTER_VECTOR_BINARY_OP`
- `LC_REGISTER_TRANSPOSE_OP`
- `LC_REGISTER_REDUCE_ROW_OP`
- `LC_REGISTER_TOPK_ROW_OP`
- `LC_REGISTER_TOPK_1D_OP`

如果你只想写：

- CPU reference
- GPU kernel / launcher
- 一行注册

那就直接用这些宏。

如果是简单 kernel，还可以用：

- `LC_REGISTER_GEMM_KERNEL`
- `LC_REGISTER_VECTOR_BINARY_KERNEL`
- `LC_REGISTER_TRANSPOSE_KERNEL`
- `LC_REGISTER_REDUCE_ROW_KERNEL`

这样连 launcher 都不用手写。
