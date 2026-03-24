#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "op_registry.hpp"
#include "utils.cuh"

namespace opfw {

using Shape = std::vector<size_t>;

enum class Stage {
    kUnknown,
    kCorrectness,
    kPerformance,
};

struct GemmShape {
    int m = 512;
    int n = 512;
    int k = 512;
};

struct MatrixShape {
    int m = 1024;
    int n = 1024;
};

struct ReduceShape {
    int m = 1024;
    int k = 4096;
};

struct TopkRowShape {
    int m = 512;
    int n = 1024;
    int k = 16;
};

struct Topk1DShape {
    int n = 1 << 20;
    int k = 16;
};

struct Softmax1DShape {
    int n = 1 << 20;
};

struct NormRowShape {
    int m = 1024;
    int k = 4096;
};

struct VectorShape {
    size_t n = static_cast<size_t>(1) << 20;
};

struct GemmSpec {
    GemmShape shape{};
    GemmShape correctness_max_shape{};
    double threshold = 1e-2;
    int warmup = 2;
    int iters = 10;
    std::vector<std::string> baselines = {"cublas_gemm_f32"};
    std::string description = "C[M,N] = A[M,K] * B[K,N]";
    std::string note = "FLOPs = 2*M*N*K";
};

struct VectorBinarySpec {
    VectorShape shape{};
    VectorShape correctness_max_shape{};
    double threshold = 1e-5;
    int warmup = 20;
    int iters = 50;
    std::vector<std::string> baselines;
    std::string description = "C[N] = A[N] (+) B[N]";
    std::string note = "bytes = read(A)+read(B)+write(C)";
};

struct TransposeSpec {
    MatrixShape shape{};
    MatrixShape correctness_max_shape{};
    double threshold = 1e-5;
    int warmup = 5;
    int iters = 20;
    std::vector<std::string> baselines;
    std::string description = "B[N,M] = transpose(A[M,N])";
    std::string note = "bytes = read(A)+write(B)";
};

struct ReduceRowSpec {
    ReduceShape shape{};
    ReduceShape correctness_max_shape{};
    double threshold = 1e-2;
    int warmup = 5;
    int iters = 20;
    std::vector<std::string> baselines;
    std::string description = "Y[M] = row_sum(A[M,K])";
    std::string note = "bytes = read(A)+write(Y)";
};

struct TopkRowSpec {
    TopkRowShape shape{};
    TopkRowShape correctness_max_shape{};
    int max_n = 0;
    double threshold = 1e-5;
    int warmup = 3;
    int iters = 20;
    std::vector<std::string> baselines;
    std::string description = "topk per row: A[M,N] -> values[M,K], indices[M,K]";
    std::string note = "bytes = read(A)+write(values)+write(indices)";
};

struct Topk1DSpec {
    Topk1DShape shape{};
    Topk1DShape correctness_max_shape{};
    double threshold = 1e-5;
    int warmup = 3;
    int iters = 20;
    std::vector<std::string> baselines;
    std::string description = "topk 1D: A[N] -> values[K], indices[K]";
    std::string note = "bytes = read(A)+write(values)+write(indices)";
};

struct Softmax1DSpec {
    Softmax1DShape shape{};
    Softmax1DShape correctness_max_shape{};
    double threshold = 1e-5;
    int warmup = 5;
    int iters = 20;
    std::vector<std::string> baselines;
    std::string description = "softmax 1D: Y[N] = softmax(X[N])";
    std::string note = "bytes = read(X)+write(Y)";
};

struct NormRowSpec {
    NormRowShape shape{};
    NormRowShape correctness_max_shape{};
    double threshold = 1e-4;
    int warmup = 5;
    int iters = 20;
    std::vector<std::string> baselines;
    std::string description = "Y[M,K] = norm_row(X[M,K])";
    std::string note = "bytes = read(X)+write(Y)";
};

using GemmCpuRef = void (*)(const float *a, const float *b, float *c, int m, int n, int k);
using GemmGpuLauncher = void (*)(float *d_c, const float *d_a, const float *d_b, int m, int n, int k);

using VectorBinaryCpuRef = void (*)(const float *a, const float *b, float *c, size_t n);
using VectorBinaryGpuLauncher = void (*)(const float *d_a, const float *d_b, float *d_c, size_t n);

using TransposeCpuRef = void (*)(const float *a, float *b, int m, int n);
using TransposeGpuLauncher = void (*)(float *d_out, const float *d_in, int m, int n);

using ReduceRowCpuRef = void (*)(const float *a, float *y, int m, int k);
using ReduceRowGpuLauncher = void (*)(float *d_y, const float *d_a, int m, int k);

using TopkRowCpuRef = void (*)(const float *a, float *out_values, int *out_indices, int m, int n, int k);
using TopkRowGpuLauncher = void (*)(float *d_out_values, int *d_out_indices, const float *d_in, int m, int n, int k);

using Topk1DCpuRef = void (*)(const float *a, int n, int k, float *out_values, int *out_indices);
using Topk1DGpuLauncher = void (*)(float *d_out_values, int *d_out_indices, const float *d_in, int n, int k);

using Softmax1DCpuRef = void (*)(const float *input, float *output, int n);
using Softmax1DGpuLauncher = void (*)(const float *d_input, float *d_output, int n);

using NormRowCpuRef = void (*)(const float *a, float *y, int m, int k);
using NormRowGpuLauncher = void (*)(float *d_y, const float *d_a, int m, int k);

Stage current_stage();
std::string current_case_name();

int current_param_int(const std::string &key, int default_value);
size_t current_param_size_t(const std::string &key, size_t default_value);

void gpu_cublas_gemm_f32(float *d_c, const float *d_a, const float *d_b, int m, int n, int k);

void cpu_gemm_f32(const float *a, const float *b, float *c, int m, int n, int k);
void cpu_vector_add_f32(const float *a, const float *b, float *c, size_t n);
void cpu_transpose_f32(const float *a, float *b, int m, int n);
void cpu_reduce_row_sum_f32(const float *a, float *y, int m, int k);
void cpu_topk_row_f32(const float *a, float *out_values, int *out_indices, int m, int n, int k);
void cpu_topk_1d_f32(const float *a, int n, int k, float *out_values, int *out_indices);
void cpu_softmax_1d_f32(const float *input, float *output, int n);

CorrectnessResult run_gemm_correctness(GemmCpuRef cpu, GemmGpuLauncher gpu, const GemmSpec &spec);
PerfResult run_gemm_performance(GemmGpuLauncher gpu, const GemmSpec &spec);

CorrectnessResult run_vector_binary_correctness(VectorBinaryCpuRef cpu, VectorBinaryGpuLauncher gpu, const VectorBinarySpec &spec);
PerfResult run_vector_binary_performance(VectorBinaryGpuLauncher gpu, const VectorBinarySpec &spec);

CorrectnessResult run_transpose_correctness(TransposeCpuRef cpu, TransposeGpuLauncher gpu, const TransposeSpec &spec);
PerfResult run_transpose_performance(TransposeGpuLauncher gpu, const TransposeSpec &spec);

CorrectnessResult run_reduce_row_correctness(ReduceRowCpuRef cpu, ReduceRowGpuLauncher gpu, const ReduceRowSpec &spec);
PerfResult run_reduce_row_performance(ReduceRowGpuLauncher gpu, const ReduceRowSpec &spec);

CorrectnessResult run_topk_row_correctness(TopkRowCpuRef cpu, TopkRowGpuLauncher gpu, const TopkRowSpec &spec);
PerfResult run_topk_row_performance(TopkRowGpuLauncher gpu, const TopkRowSpec &spec);

CorrectnessResult run_topk_1d_correctness(Topk1DCpuRef cpu, Topk1DGpuLauncher gpu, const Topk1DSpec &spec);
PerfResult run_topk_1d_performance(Topk1DGpuLauncher gpu, const Topk1DSpec &spec);

CorrectnessResult run_softmax_1d_correctness(Softmax1DCpuRef cpu, Softmax1DGpuLauncher gpu, const Softmax1DSpec &spec);
PerfResult run_softmax_1d_performance(Softmax1DGpuLauncher gpu, const Softmax1DSpec &spec);

CorrectnessResult run_norm_row_correctness(NormRowCpuRef cpu, NormRowGpuLauncher gpu, const NormRowSpec &spec);
PerfResult run_norm_row_performance(NormRowGpuLauncher gpu, const NormRowSpec &spec);

OpEntry make_gemm_op(const char *name, GemmCpuRef cpu, GemmGpuLauncher gpu, GemmSpec spec = {});
OpEntry make_vector_binary_op(const char *name, VectorBinaryCpuRef cpu, VectorBinaryGpuLauncher gpu, VectorBinarySpec spec = {});
OpEntry make_transpose_op(const char *name, TransposeCpuRef cpu, TransposeGpuLauncher gpu, TransposeSpec spec = {});
OpEntry make_reduce_row_op(const char *name, ReduceRowCpuRef cpu, ReduceRowGpuLauncher gpu, ReduceRowSpec spec = {});
OpEntry make_topk_row_op(const char *name, TopkRowCpuRef cpu, TopkRowGpuLauncher gpu, TopkRowSpec spec = {});
OpEntry make_topk_1d_op(const char *name, Topk1DCpuRef cpu, Topk1DGpuLauncher gpu, Topk1DSpec spec = {});
OpEntry make_softmax_1d_op(const char *name, Softmax1DCpuRef cpu, Softmax1DGpuLauncher gpu, Softmax1DSpec spec = {});
OpEntry make_norm_row_op(const char *name, NormRowCpuRef cpu, NormRowGpuLauncher gpu, NormRowSpec spec = {});

} // namespace opfw

#define LC_REGISTER_GEMM_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_gemm_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_GEMM_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_gemm_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_VECTOR_BINARY_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_vector_binary_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_VECTOR_BINARY_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_vector_binary_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_TRANSPOSE_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_transpose_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_TRANSPOSE_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_transpose_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_REDUCE_ROW_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_reduce_row_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_REDUCE_ROW_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_reduce_row_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_TOPK_ROW_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_topk_row_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_TOPK_ROW_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_topk_row_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_TOPK_1D_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_topk_1d_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_TOPK_1D_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_topk_1d_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_SOFTMAX_1D_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_softmax_1d_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_SOFTMAX_1D_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_softmax_1d_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_NORM_ROW_OP(NAME, CPU, GPU) OPFW_REGISTER_ENTRY(opfw::make_norm_row_op((NAME), (CPU), (GPU)))
#define LC_REGISTER_NORM_ROW_OP_EX(NAME, CPU, GPU, SPEC) OPFW_REGISTER_ENTRY(opfw::make_norm_row_op((NAME), (CPU), (GPU), (SPEC)))

#define LC_REGISTER_VECTOR_BINARY_KERNEL(NAME, CPU, KERNEL, BLOCK_SIZE)                                                                                         \
    namespace {                                                                                                                                                \
    static void OPFW_CONCAT(_lc_vector_launcher_, __LINE__)(const float *d_a, const float *d_b, float *d_c, size_t n) {                                     \
        const int block = (BLOCK_SIZE);                                                                                                                       \
        const int grid = static_cast<int>((n + static_cast<size_t>(block) - 1) / static_cast<size_t>(block));                                               \
        KERNEL<<<grid, block>>>(d_a, d_b, d_c, n);                                                                                                            \
        CUDA_CHECK(cudaGetLastError());                                                                                                                       \
    }                                                                                                                                                          \
    }                                                                                                                                                          \
    LC_REGISTER_VECTOR_BINARY_OP((NAME), (CPU), OPFW_CONCAT(_lc_vector_launcher_, __LINE__))

#define LC_REGISTER_GEMM_KERNEL(NAME, CPU, KERNEL, BLOCK_X, BLOCK_Y)                                                                                           \
    namespace {                                                                                                                                                \
    static void OPFW_CONCAT(_lc_gemm_launcher_, __LINE__)(float *d_c, const float *d_a, const float *d_b, int m, int n, int k) {                           \
        dim3 block((BLOCK_X), (BLOCK_Y));                                                                                                                     \
        dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);                                                                                 \
        KERNEL<<<grid, block>>>(d_a, d_b, d_c, m, n, k);                                                                                                      \
        CUDA_CHECK(cudaGetLastError());                                                                                                                       \
    }                                                                                                                                                          \
    }                                                                                                                                                          \
    LC_REGISTER_GEMM_OP((NAME), (CPU), OPFW_CONCAT(_lc_gemm_launcher_, __LINE__))

#define LC_REGISTER_TRANSPOSE_KERNEL(NAME, CPU, KERNEL, BLOCK_X, BLOCK_Y)                                                                                      \
    namespace {                                                                                                                                                \
    static void OPFW_CONCAT(_lc_transpose_launcher_, __LINE__)(float *d_out, const float *d_in, int m, int n) {                                             \
        dim3 block((BLOCK_X), (BLOCK_Y));                                                                                                                     \
        dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);                                                                                 \
        KERNEL<<<grid, block>>>(d_in, d_out, m, n);                                                                                                           \
        CUDA_CHECK(cudaGetLastError());                                                                                                                       \
    }                                                                                                                                                          \
    }                                                                                                                                                          \
    LC_REGISTER_TRANSPOSE_OP((NAME), (CPU), OPFW_CONCAT(_lc_transpose_launcher_, __LINE__))

#define LC_REGISTER_REDUCE_ROW_KERNEL(NAME, CPU, KERNEL, THREADS)                                                                                              \
    namespace {                                                                                                                                                \
    static void OPFW_CONCAT(_lc_reduce_launcher_, __LINE__)(float *d_y, const float *d_a, int m, int k) {                                                   \
        dim3 block((THREADS));                                                                                                                                 \
        dim3 grid(m);                                                                                                                                          \
        KERNEL<<<grid, block>>>(d_a, d_y, m, k);                                                                                                              \
        CUDA_CHECK(cudaGetLastError());                                                                                                                       \
    }                                                                                                                                                          \
    }                                                                                                                                                          \
    LC_REGISTER_REDUCE_ROW_OP((NAME), (CPU), OPFW_CONCAT(_lc_reduce_launcher_, __LINE__))

#define LC_REGISTER_NORM_ROW_KERNEL(NAME, CPU, KERNEL, THREADS)                                                                                                \
    namespace {                                                                                                                                                \
    static void OPFW_CONCAT(_lc_norm_launcher_, __LINE__)(float *d_y, const float *d_a, int m, int k) {                                                     \
        dim3 block((THREADS));                                                                                                                                 \
        dim3 grid(m);                                                                                                                                          \
        KERNEL<<<grid, block>>>(d_a, d_y, m, k);                                                                                                              \
        CUDA_CHECK(cudaGetLastError());                                                                                                                       \
    }                                                                                                                                                          \
    }                                                                                                                                                          \
    LC_REGISTER_NORM_ROW_OP((NAME), (CPU), OPFW_CONCAT(_lc_norm_launcher_, __LINE__))
