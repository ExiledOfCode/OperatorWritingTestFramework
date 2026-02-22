# 测试框架说明

本仓库用于对 CUDA 算子进行 **正确性（Correctness）** 与 **性能（Performance）** 测试，整体基于 GoogleTest，并提供运行与调试脚本。

---

## 目录结构

- `test_op_name.sh`  
  存放需要进行测试的算子名称列表（用于选择/过滤测试目标）。

- `tests/`  
  GoogleTest 测试用例目录（各算子的单元测试、正确性验证、基准测试等）。

- `tool/`  
  GPU 参数检测相关脚本（用于获取当前 GPU 型号、算力、SM 数量等信息，便于测试配置/输出）。

- `include/`  
  测试框架封装代码（例如：计时器、Kernel launch 封装、公共工具函数、索引计算封装等）。

- `src/`  
  算子实现目录  
  - `src/gemm/`
    - `hand_idx/`：纯手写 idx 计算版本
    - `frame_idx/`：使用框架封装的 idx 计算版本
  - `src/reduce/`
    - Reduce 类算子相关实现

---

## 运行方式

### 1) 正确性 + 性能测试
```sh
./run.sh