#include "op_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cublas_v2.h>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace opfw {
namespace {

std::string normalize_key(const std::string &key) {
    std::string normalized;
    normalized.reserve(key.size());
    for (unsigned char ch : key) {
        if (std::isalnum(ch) != 0) {
            normalized.push_back(static_cast<char>(std::toupper(ch)));
        } else {
            normalized.push_back('_');
        }
    }
    return normalized;
}

std::string param_env_key(const std::string &key) {
    return "CUDA_OP_PARAM_" + normalize_key(key);
}

template <typename T>
T parse_integer_or_default(const char *raw, T default_value) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }

    char *end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
        return default_value;
    }
    return static_cast<T>(parsed);
}

int env_int(const std::string &key, int default_value) {
    return parse_integer_or_default<int>(std::getenv(key.c_str()), default_value);
}

size_t env_size_t(const std::string &key, size_t default_value) {
    return parse_integer_or_default<size_t>(std::getenv(key.c_str()), default_value);
}

int current_seed(int default_value = 123) {
    return env_int("CUDA_OP_SEED", default_value);
}

int current_warmup(int default_value) {
    return std::max(0, env_int("CUDA_OP_WARMUP", default_value));
}

int current_iters(int default_value) {
    return std::max(1, env_int("CUDA_OP_ITERS", default_value));
}

void fail_runtime(const std::string &message) {
    std::cerr << "[opfw] " << message << std::endl;
    std::exit(EXIT_FAILURE);
}

size_t shape_numel(const Shape &shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

size_t bytes_for_shapes(const std::vector<Shape> &shapes, size_t bytes_per_element) {
    size_t total = 0;
    for (const auto &shape : shapes) {
        total += shape_numel(shape) * bytes_per_element;
    }
    return total;
}

CorrectnessResult make_correctness_result(double metric, const std::string &metric_name, double threshold, const std::string &note = "") {
    return CorrectnessResult{metric <= threshold, metric, metric_name, threshold, note};
}

PerfResult make_perf_result(double ms, double cpu_ms, const std::string &unit_name, double unit_value, const std::string &note, size_t input_bytes,
                            size_t output_bytes, const std::vector<Shape> &input_shapes, const std::vector<Shape> &output_shapes) {
    PerfResult result;
    result.ms = ms;
    result.cpu_ms = cpu_ms;
    result.unit_name = unit_name;
    result.unit_value = unit_value;
    result.note = note;
    result.input_size = input_bytes;
    result.output_size = output_bytes;
    result.input_format = input_shapes;
    result.output_format = output_shapes;
    return result;
}

void add_metric(PerfResult *result, const std::string &name, double value, const std::string &unit) {
    result->extra_metrics.push_back(MetricEntry{name, value, unit});
}

std::string append_note(const std::string &base, const std::string &extra) {
    if (extra.empty()) {
        return base;
    }
    if (base.empty()) {
        return extra;
    }
    return base + "; " + extra;
}

bool contains_name(const std::vector<std::string> &values, const std::string &target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}

template <typename T>
void fill_uniform(std::vector<T> &values, int seed, T low, T high) {
    std::mt19937 gen(seed);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<long long> dist(static_cast<long long>(low), static_cast<long long>(high));
        for (auto &value : values) {
            value = static_cast<T>(dist(gen));
        }
    } else {
        std::uniform_real_distribution<double> dist(static_cast<double>(low), static_cast<double>(high));
        for (auto &value : values) {
            value = static_cast<T>(dist(gen));
        }
    }
}

double max_abs_diff(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
    }
    return max_abs;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

bool shape_equals(const GemmShape &lhs, const GemmShape &rhs) {
    return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k;
}

std::string format_shape(const GemmShape &shape) {
    std::ostringstream oss;
    oss << "M=" << shape.m << ",N=" << shape.n << ",K=" << shape.k;
    return oss.str();
}

GemmShape gemm_correctness_limit(const GemmSpec &spec) {
    return GemmShape{
        spec.correctness_max_shape.m > 0 ? spec.correctness_max_shape.m : spec.shape.m,
        spec.correctness_max_shape.n > 0 ? spec.correctness_max_shape.n : spec.shape.n,
        spec.correctness_max_shape.k > 0 ? spec.correctness_max_shape.k : spec.shape.k,
    };
}

GemmShape clamp_shape(const GemmShape &requested, const GemmShape &limit) {
    return GemmShape{
        std::min(requested.m, limit.m),
        std::min(requested.n, limit.n),
        std::min(requested.k, limit.k),
    };
}

bool shape_equals(const VectorShape &lhs, const VectorShape &rhs) {
    return lhs.n == rhs.n;
}

std::string format_shape(const VectorShape &shape) {
    std::ostringstream oss;
    oss << "N=" << shape.n;
    return oss.str();
}

VectorShape vector_correctness_limit(const VectorBinarySpec &spec) {
    return VectorShape{
        spec.correctness_max_shape.n > 0 ? spec.correctness_max_shape.n : spec.shape.n,
    };
}

VectorShape clamp_shape(const VectorShape &requested, const VectorShape &limit) {
    return VectorShape{
        std::min(requested.n, limit.n),
    };
}

bool shape_equals(const MatrixShape &lhs, const MatrixShape &rhs) {
    return lhs.m == rhs.m && lhs.n == rhs.n;
}

std::string format_shape(const MatrixShape &shape) {
    std::ostringstream oss;
    oss << "M=" << shape.m << ",N=" << shape.n;
    return oss.str();
}

MatrixShape matrix_correctness_limit(const TransposeSpec &spec) {
    return MatrixShape{
        spec.correctness_max_shape.m > 0 ? spec.correctness_max_shape.m : spec.shape.m,
        spec.correctness_max_shape.n > 0 ? spec.correctness_max_shape.n : spec.shape.n,
    };
}

MatrixShape clamp_shape(const MatrixShape &requested, const MatrixShape &limit) {
    return MatrixShape{
        std::min(requested.m, limit.m),
        std::min(requested.n, limit.n),
    };
}

bool shape_equals(const ReduceShape &lhs, const ReduceShape &rhs) {
    return lhs.m == rhs.m && lhs.k == rhs.k;
}

std::string format_shape(const ReduceShape &shape) {
    std::ostringstream oss;
    oss << "M=" << shape.m << ",K=" << shape.k;
    return oss.str();
}

ReduceShape reduce_correctness_limit(const ReduceRowSpec &spec) {
    return ReduceShape{
        spec.correctness_max_shape.m > 0 ? spec.correctness_max_shape.m : spec.shape.m,
        spec.correctness_max_shape.k > 0 ? spec.correctness_max_shape.k : spec.shape.k,
    };
}

ReduceShape clamp_shape(const ReduceShape &requested, const ReduceShape &limit) {
    return ReduceShape{
        std::min(requested.m, limit.m),
        std::min(requested.k, limit.k),
    };
}

bool shape_equals(const TopkRowShape &lhs, const TopkRowShape &rhs) {
    return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k;
}

std::string format_shape(const TopkRowShape &shape) {
    std::ostringstream oss;
    oss << "M=" << shape.m << ",N=" << shape.n << ",K=" << shape.k;
    return oss.str();
}

TopkRowShape topk_row_correctness_limit(const TopkRowSpec &spec) {
    return TopkRowShape{
        spec.correctness_max_shape.m > 0 ? spec.correctness_max_shape.m : spec.shape.m,
        spec.correctness_max_shape.n > 0 ? spec.correctness_max_shape.n : spec.shape.n,
        spec.correctness_max_shape.k > 0 ? spec.correctness_max_shape.k : spec.shape.k,
    };
}

TopkRowShape clamp_shape(const TopkRowShape &requested, const TopkRowShape &limit) {
    return TopkRowShape{
        std::min(requested.m, limit.m),
        std::min(requested.n, limit.n),
        std::min(requested.k, limit.k),
    };
}

bool shape_equals(const Topk1DShape &lhs, const Topk1DShape &rhs) {
    return lhs.n == rhs.n && lhs.k == rhs.k;
}

std::string format_shape(const Topk1DShape &shape) {
    std::ostringstream oss;
    oss << "N=" << shape.n << ",K=" << shape.k;
    return oss.str();
}

Topk1DShape topk_1d_correctness_limit(const Topk1DSpec &spec) {
    return Topk1DShape{
        spec.correctness_max_shape.n > 0 ? spec.correctness_max_shape.n : spec.shape.n,
        spec.correctness_max_shape.k > 0 ? spec.correctness_max_shape.k : spec.shape.k,
    };
}

Topk1DShape clamp_shape(const Topk1DShape &requested, const Topk1DShape &limit) {
    return Topk1DShape{
        std::min(requested.n, limit.n),
        std::min(requested.k, limit.k),
    };
}

bool shape_equals(const Softmax1DShape &lhs, const Softmax1DShape &rhs) {
    return lhs.n == rhs.n;
}

std::string format_shape(const Softmax1DShape &shape) {
    std::ostringstream oss;
    oss << "N=" << shape.n;
    return oss.str();
}

Softmax1DShape softmax_1d_correctness_limit(const Softmax1DSpec &spec) {
    return Softmax1DShape{
        spec.correctness_max_shape.n > 0 ? spec.correctness_max_shape.n : spec.shape.n,
    };
}

Softmax1DShape clamp_shape(const Softmax1DShape &requested, const Softmax1DShape &limit) {
    return Softmax1DShape{
        std::min(requested.n, limit.n),
    };
}

bool shape_equals(const NormRowShape &lhs, const NormRowShape &rhs) {
    return lhs.m == rhs.m && lhs.k == rhs.k;
}

std::string format_shape(const NormRowShape &shape) {
    std::ostringstream oss;
    oss << "M=" << shape.m << ",K=" << shape.k;
    return oss.str();
}

NormRowShape norm_row_correctness_limit(const NormRowSpec &spec) {
    return NormRowShape{
        spec.correctness_max_shape.m > 0 ? spec.correctness_max_shape.m : spec.shape.m,
        spec.correctness_max_shape.k > 0 ? spec.correctness_max_shape.k : spec.shape.k,
    };
}

NormRowShape clamp_shape(const NormRowShape &requested, const NormRowShape &limit) {
    return NormRowShape{
        std::min(requested.m, limit.m),
        std::min(requested.k, limit.k),
    };
}

GemmShape current_gemm_shape(const GemmSpec &spec) {
    return GemmShape{
        current_param_int("M", spec.shape.m),
        current_param_int("N", spec.shape.n),
        current_param_int("K", spec.shape.k),
    };
}

VectorShape current_vector_shape(const VectorBinarySpec &spec) {
    return VectorShape{current_param_size_t("N", spec.shape.n)};
}

MatrixShape current_matrix_shape(const TransposeSpec &spec) {
    return MatrixShape{
        current_param_int("M", spec.shape.m),
        current_param_int("N", spec.shape.n),
    };
}

ReduceShape current_reduce_shape(const ReduceRowSpec &spec) {
    return ReduceShape{
        current_param_int("M", spec.shape.m),
        current_param_int("K", spec.shape.k),
    };
}

TopkRowShape current_topk_row_shape(const TopkRowSpec &spec) {
    TopkRowShape shape{
        current_param_int("M", spec.shape.m),
        current_param_int("N", spec.shape.n),
        current_param_int("K", spec.shape.k),
    };
    if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0) {
        fail_runtime("topk_row requires M > 0, N > 0, K > 0.");
    }
    if (shape.k > shape.n) {
        fail_runtime("topk_row requires K <= N.");
    }
    if (spec.max_n > 0 && shape.n > spec.max_n) {
        fail_runtime("topk_row shape exceeds registered max N.");
    }
    return shape;
}

Topk1DShape current_topk_1d_shape(const Topk1DSpec &spec) {
    Topk1DShape shape{
        current_param_int("N", spec.shape.n),
        current_param_int("K", spec.shape.k),
    };
    if (shape.n <= 0 || shape.k <= 0) {
        fail_runtime("topk_1d requires N > 0, K > 0.");
    }
    if (shape.k > shape.n) {
        fail_runtime("topk_1d requires K <= N.");
    }
    return shape;
}

Softmax1DShape current_softmax_1d_shape(const Softmax1DSpec &spec) {
    Softmax1DShape shape{
        current_param_int("N", spec.shape.n),
    };
    if (shape.n <= 0) {
        fail_runtime("softmax_1d requires N > 0.");
    }
    return shape;
}

NormRowShape current_norm_row_shape(const NormRowSpec &spec) {
    NormRowShape shape{
        current_param_int("M", spec.shape.m),
        current_param_int("K", spec.shape.k),
    };
    if (shape.m <= 0 || shape.k <= 0) {
        fail_runtime("norm_row requires M > 0, K > 0.");
    }
    return shape;
}

OpMetadata build_metadata(const std::string &name, const std::string &kind, const std::vector<ParamSpec> &params, const std::vector<std::string> &edge_axes,
                          const std::vector<std::string> &baselines, const std::string &description) {
    OpMetadata meta;
    meta.name = name;
    meta.kind = kind;
    meta.params = params;
    meta.edge_axes = edge_axes;
    meta.baselines = baselines;
    meta.description = description;
    return meta;
}

} // namespace

Stage current_stage() {
    const char *raw = std::getenv("CUDA_OP_STAGE");
    if (raw == nullptr) {
        return Stage::kUnknown;
    }
    const std::string stage(raw);
    if (stage == "correctness") {
        return Stage::kCorrectness;
    }
    if (stage == "performance") {
        return Stage::kPerformance;
    }
    return Stage::kUnknown;
}

std::string current_case_name() {
    const char *raw = std::getenv("CUDA_OP_CASE_NAME");
    return raw == nullptr ? std::string("default_case") : std::string(raw);
}

int current_param_int(const std::string &key, int default_value) {
    return env_int(param_env_key(key), default_value);
}

size_t current_param_size_t(const std::string &key, size_t default_value) {
    return env_size_t(param_env_key(key), default_value);
}

void gpu_cublas_gemm_f32(float *d_c, const float *d_a, const float *d_b, int m, int n, int k) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasCreate failed, status = " << int(st) << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    /*m=*/n, /*n=*/m, /*k=*/k, &alpha,
                                    /*A=*/d_b, /*lda=*/n,
                                    /*B=*/d_a, /*ldb=*/k,
                                    &beta,
                                    /*C=*/d_c, /*ldc=*/n);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed, status = " << int(st) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

void cpu_gemm_f32(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

void cpu_vector_add_f32(const float *a, const float *b, float *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void cpu_transpose_f32(const float *a, float *b, int m, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            b[col * m + row] = a[row * n + col];
        }
    }
}

void cpu_reduce_row_sum_f32(const float *a, float *y, int m, int k) {
    for (int row = 0; row < m; ++row) {
        float acc = 0.0f;
        for (int col = 0; col < k; ++col) {
            acc += a[row * k + col];
        }
        y[row] = acc;
    }
}

void cpu_topk_row_f32(const float *a, float *out_values, int *out_indices, int m, int n, int k) {
    std::vector<std::pair<float, int>> tmp;
    tmp.reserve(static_cast<size_t>(n));

    for (int row = 0; row < m; ++row) {
        tmp.clear();
        const float *row_ptr = a + static_cast<size_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            tmp.emplace_back(row_ptr[col], col);
        }

        if (k < n) {
            std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end(), [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
            std::sort(tmp.begin(), tmp.begin() + k, [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
        } else {
            std::sort(tmp.begin(), tmp.end(), [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
        }

        for (int i = 0; i < k; ++i) {
            out_values[static_cast<size_t>(row) * k + i] = tmp[i].first;
            out_indices[static_cast<size_t>(row) * k + i] = tmp[i].second;
        }
    }
}

void cpu_topk_1d_f32(const float *a, int n, int k, float *out_values, int *out_indices) {
    std::vector<std::pair<float, int>> tmp;
    tmp.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        tmp.emplace_back(a[i], i);
    }
    if (k < n) {
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end(), [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
        std::sort(tmp.begin(), tmp.begin() + k, [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    } else {
        std::sort(tmp.begin(), tmp.end(), [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    }
    for (int i = 0; i < k; ++i) {
        out_values[i] = tmp[i].first;
        out_indices[i] = tmp[i].second;
    }
}

void cpu_softmax_1d_f32(const float *input, float *output, int n) {
    float max_value = input[0];
    for (int i = 1; i < n; ++i) {
        max_value = std::max(max_value, input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }

    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

CorrectnessResult run_gemm_correctness(GemmCpuRef cpu, GemmGpuLauncher gpu, const GemmSpec &spec) {
    const GemmShape requested_shape = current_gemm_shape(spec);
    const bool use_cublas_reference = contains_name(spec.baselines, "cublas_gemm_f32");
    const GemmShape shape = use_cublas_reference ? requested_shape : clamp_shape(requested_shape, gemm_correctness_limit(spec));
    std::vector<float> h_a(static_cast<size_t>(shape.m) * shape.k);
    std::vector<float> h_b(static_cast<size_t>(shape.k) * shape.n);
    std::vector<float> h_ref(static_cast<size_t>(shape.m) * shape.n);
    std::vector<float> h_out(static_cast<size_t>(shape.m) * shape.n);
    std::string note = use_cublas_reference ? "reference=cublas_gemm_f32" : "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    const int seed = current_seed();
    fill_uniform(h_a, seed, -1.0f, 1.0f);
    fill_uniform(h_b, seed + 1, -1.0f, 1.0f);

    DeviceBuffer<float> d_a(h_a.size());
    DeviceBuffer<float> d_b(h_b.size());
    DeviceBuffer<float> d_ref(h_ref.size());
    DeviceBuffer<float> d_c(h_out.size());

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    if (use_cublas_reference) {
        gpu_cublas_gemm_f32(d_ref.data(), d_a.data(), d_b.data(), shape.m, shape.n, shape.k);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref.data(), h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        cpu(h_a.data(), h_b.data(), h_ref.data(), shape.m, shape.n, shape.k);
    }

    gpu(d_c.data(), d_a.data(), d_b.data(), shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c.data(), h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return make_correctness_result(max_abs_diff(h_ref, h_out), "max_abs_diff", spec.threshold, note);
}

PerfResult run_gemm_performance(GemmGpuLauncher gpu, const GemmSpec &spec) {
    const GemmShape shape = current_gemm_shape(spec);
    std::vector<float> h_a(static_cast<size_t>(shape.m) * shape.k, 1.0f);
    std::vector<float> h_b(static_cast<size_t>(shape.k) * shape.n, 1.0f);

    DeviceBuffer<float> d_a(h_a.size());
    DeviceBuffer<float> d_b(h_b.size());
    DeviceBuffer<float> d_c(static_cast<size_t>(shape.m) * shape.n);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_c.data(), d_a.data(), d_b.data(), shape.m, shape.n, shape.k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_c.data(), d_a.data(), d_b.data(), shape.m, shape.n, shape.k);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const double flops = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) * static_cast<double>(shape.k);
    const double tflops = flops / (ms / 1000.0) / 1.0e12;
    const size_t input_bytes = (static_cast<size_t>(shape.m) * shape.k + static_cast<size_t>(shape.k) * shape.n) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(shape.m) * shape.n * sizeof(float);

    auto result = make_perf_result(ms, 0.0, "TFLOP/s", tflops, spec.note, input_bytes, output_bytes,
                                   {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)},
                                    {static_cast<size_t>(shape.k), static_cast<size_t>(shape.n)}},
                                   {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.n)}});
    add_metric(&result, "effective_bandwidth", static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9, "GB/s");
    return result;
}

CorrectnessResult run_vector_binary_correctness(VectorBinaryCpuRef cpu, VectorBinaryGpuLauncher gpu, const VectorBinarySpec &spec) {
    const VectorShape requested_shape = current_vector_shape(spec);
    const VectorShape shape = clamp_shape(requested_shape, vector_correctness_limit(spec));
    const size_t n = shape.n;
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_out(n);
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    const int seed = current_seed();
    fill_uniform(h_a, seed, -1.0f, 1.0f);
    fill_uniform(h_b, seed + 1, -1.0f, 1.0f);

    cpu(h_a.data(), h_b.data(), h_ref.data(), n);

    DeviceBuffer<float> d_a(n);
    DeviceBuffer<float> d_b(n);
    DeviceBuffer<float> d_c(n);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    gpu(d_a.data(), d_b.data(), d_c.data(), n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c.data(), n * sizeof(float), cudaMemcpyDeviceToHost));

    return make_correctness_result(max_abs_diff(h_ref, h_out), "max_abs_diff", spec.threshold, note);
}

PerfResult run_vector_binary_performance(VectorBinaryGpuLauncher gpu, const VectorBinarySpec &spec) {
    const size_t n = current_vector_shape(spec).n;
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);

    DeviceBuffer<float> d_a(n);
    DeviceBuffer<float> d_b(n);
    DeviceBuffer<float> d_c(n);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_a.data(), d_b.data(), d_c.data(), n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_a.data(), d_b.data(), d_c.data(), n);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = 2 * n * sizeof(float);
    const size_t output_bytes = n * sizeof(float);
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;

    auto result = make_perf_result(ms, 0.0, "GB/s", gbps, spec.note, input_bytes, output_bytes, {{n}, {n}}, {{n}});
    add_metric(&result, "elements_per_second", static_cast<double>(n) / (ms / 1000.0) / 1.0e9, "Gelem/s");
    return result;
}

CorrectnessResult run_transpose_correctness(TransposeCpuRef cpu, TransposeGpuLauncher gpu, const TransposeSpec &spec) {
    const MatrixShape requested_shape = current_matrix_shape(spec);
    const MatrixShape shape = clamp_shape(requested_shape, matrix_correctness_limit(spec));
    const size_t count = static_cast<size_t>(shape.m) * shape.n;
    std::vector<float> h_in(count);
    std::vector<float> h_ref(count);
    std::vector<float> h_out(count);
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    fill_uniform(h_in, current_seed(), -1.0f, 1.0f);
    cpu(h_in.data(), h_ref.data(), shape.m, shape.n);

    DeviceBuffer<float> d_in(count);
    DeviceBuffer<float> d_out(count);

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    gpu(d_out.data(), d_in.data(), shape.m, shape.n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.data(), count * sizeof(float), cudaMemcpyDeviceToHost));

    return make_correctness_result(max_abs_diff(h_ref, h_out), "max_abs_diff", spec.threshold, note);
}

PerfResult run_transpose_performance(TransposeGpuLauncher gpu, const TransposeSpec &spec) {
    const MatrixShape shape = current_matrix_shape(spec);
    const size_t count = static_cast<size_t>(shape.m) * shape.n;
    std::vector<float> h_in(count, 1.0f);

    DeviceBuffer<float> d_in(count);
    DeviceBuffer<float> d_out(count);

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_out.data(), d_in.data(), shape.m, shape.n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_out.data(), d_in.data(), shape.m, shape.n);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = count * sizeof(float);
    const size_t output_bytes = count * sizeof(float);
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;

    return make_perf_result(ms, 0.0, "GB/s", gbps, spec.note, input_bytes, output_bytes,
                            {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.n)}},
                            {{static_cast<size_t>(shape.n), static_cast<size_t>(shape.m)}});
}

CorrectnessResult run_reduce_row_correctness(ReduceRowCpuRef cpu, ReduceRowGpuLauncher gpu, const ReduceRowSpec &spec) {
    const ReduceShape requested_shape = current_reduce_shape(spec);
    const ReduceShape shape = clamp_shape(requested_shape, reduce_correctness_limit(spec));
    std::vector<float> h_a(static_cast<size_t>(shape.m) * shape.k);
    std::vector<float> h_ref(shape.m);
    std::vector<float> h_out(shape.m);
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    fill_uniform(h_a, current_seed(), -1.0f, 1.0f);
    cpu(h_a.data(), h_ref.data(), shape.m, shape.k);

    DeviceBuffer<float> d_a(h_a.size());
    DeviceBuffer<float> d_y(h_out.size());

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu(d_y.data(), d_a.data(), shape.m, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y.data(), h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return make_correctness_result(max_abs_diff(h_ref, h_out), "max_abs_diff", spec.threshold, note);
}

PerfResult run_reduce_row_performance(ReduceRowGpuLauncher gpu, const ReduceRowSpec &spec) {
    const ReduceShape shape = current_reduce_shape(spec);
    std::vector<float> h_a(static_cast<size_t>(shape.m) * shape.k, 1.0f);

    DeviceBuffer<float> d_a(h_a.size());
    DeviceBuffer<float> d_y(shape.m);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_y.data(), d_a.data(), shape.m, shape.k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_y.data(), d_a.data(), shape.m, shape.k);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = h_a.size() * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(shape.m) * sizeof(float);
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;

    return make_perf_result(ms, 0.0, "GB/s", gbps, spec.note, input_bytes, output_bytes,
                            {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)}}, {{static_cast<size_t>(shape.m)}});
}

CorrectnessResult run_topk_row_correctness(TopkRowCpuRef cpu, TopkRowGpuLauncher gpu, const TopkRowSpec &spec) {
    const TopkRowShape requested_shape = current_topk_row_shape(spec);
    const TopkRowShape shape = clamp_shape(requested_shape, topk_row_correctness_limit(spec));
    std::vector<float> h_in(static_cast<size_t>(shape.m) * shape.n);
    std::vector<float> h_ref_values(static_cast<size_t>(shape.m) * shape.k);
    std::vector<int> h_ref_indices(static_cast<size_t>(shape.m) * shape.k);
    std::vector<float> h_out_values(static_cast<size_t>(shape.m) * shape.k);
    std::vector<int> h_out_indices(static_cast<size_t>(shape.m) * shape.k);
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    fill_uniform(h_in, current_seed(), -1.0f, 1.0f);
    cpu(h_in.data(), h_ref_values.data(), h_ref_indices.data(), shape.m, shape.n, shape.k);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out_values(h_out_values.size());
    DeviceBuffer<int> d_out_indices(h_out_indices.size());

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu(d_out_values.data(), d_out_indices.data(), d_in.data(), shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out_values.data(), d_out_values.data(), h_out_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_indices.data(), d_out_indices.data(), h_out_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));

    int bad_indices = 0;
    for (size_t i = 0; i < h_ref_indices.size(); ++i) {
        if (h_ref_indices[i] != h_out_indices[i]) {
            ++bad_indices;
        }
    }
    note = append_note(note, "idx_mismatch_count=" + std::to_string(bad_indices));

    return make_correctness_result(max_abs_diff(h_ref_values, h_out_values), "max_abs_diff", spec.threshold, note);
}

PerfResult run_topk_row_performance(TopkRowGpuLauncher gpu, const TopkRowSpec &spec) {
    const TopkRowShape shape = current_topk_row_shape(spec);
    std::vector<float> h_in(static_cast<size_t>(shape.m) * shape.n, 1.0f);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out_values(static_cast<size_t>(shape.m) * shape.k);
    DeviceBuffer<int> d_out_indices(static_cast<size_t>(shape.m) * shape.k);

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_out_values.data(), d_out_indices.data(), d_in.data(), shape.m, shape.n, shape.k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_out_values.data(), d_out_indices.data(), d_in.data(), shape.m, shape.n, shape.k);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = h_in.size() * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(shape.m) * shape.k * (sizeof(float) + sizeof(int));
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;

    return make_perf_result(ms, 0.0, "GB/s", gbps, spec.note, input_bytes, output_bytes,
                            {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.n)}},
                            {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)},
                             {static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)}});
}

CorrectnessResult run_topk_1d_correctness(Topk1DCpuRef cpu, Topk1DGpuLauncher gpu, const Topk1DSpec &spec) {
    const Topk1DShape requested_shape = current_topk_1d_shape(spec);
    const Topk1DShape shape = clamp_shape(requested_shape, topk_1d_correctness_limit(spec));
    std::vector<float> h_in(shape.n);
    std::vector<float> h_ref_values(shape.k);
    std::vector<int> h_ref_indices(shape.k);
    std::vector<float> h_out_values(shape.k);
    std::vector<int> h_out_indices(shape.k);
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    fill_uniform(h_in, current_seed(), -1.0f, 1.0f);
    cpu(h_in.data(), shape.n, shape.k, h_ref_values.data(), h_ref_indices.data());

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out_values(h_out_values.size());
    DeviceBuffer<int> d_out_indices(h_out_indices.size());

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu(d_out_values.data(), d_out_indices.data(), d_in.data(), shape.n, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out_values.data(), d_out_values.data(), h_out_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_indices.data(), d_out_indices.data(), h_out_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));

    int bad_indices = 0;
    for (size_t i = 0; i < h_ref_indices.size(); ++i) {
        if (h_ref_indices[i] != h_out_indices[i]) {
            ++bad_indices;
        }
    }
    note = append_note(note, "idx_mismatch_count=" + std::to_string(bad_indices));

    return make_correctness_result(max_abs_diff(h_ref_values, h_out_values), "max_abs_diff", spec.threshold, note);
}

PerfResult run_topk_1d_performance(Topk1DGpuLauncher gpu, const Topk1DSpec &spec) {
    const Topk1DShape shape = current_topk_1d_shape(spec);
    std::vector<float> h_in(shape.n, 1.0f);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out_values(shape.k);
    DeviceBuffer<int> d_out_indices(shape.k);

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_out_values.data(), d_out_indices.data(), d_in.data(), shape.n, shape.k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_out_values.data(), d_out_indices.data(), d_in.data(), shape.n, shape.k);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = h_in.size() * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(shape.k) * (sizeof(float) + sizeof(int));
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;

    return make_perf_result(ms, 0.0, "GB/s", gbps, spec.note, input_bytes, output_bytes, {{static_cast<size_t>(shape.n)}},
                            {{static_cast<size_t>(shape.k)}, {static_cast<size_t>(shape.k)}});
}

CorrectnessResult run_softmax_1d_correctness(Softmax1DCpuRef cpu, Softmax1DGpuLauncher gpu, const Softmax1DSpec &spec) {
    const Softmax1DShape requested_shape = current_softmax_1d_shape(spec);
    const Softmax1DShape shape = clamp_shape(requested_shape, softmax_1d_correctness_limit(spec));
    std::vector<float> h_in(shape.n);
    std::vector<float> h_ref(shape.n);
    std::vector<float> h_out(shape.n);
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    fill_uniform(h_in, current_seed(), -4.0f, 4.0f);
    cpu(h_in.data(), h_ref.data(), shape.n);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out(h_out.size());

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu(d_in.data(), d_out.data(), shape.n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.data(), h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return make_correctness_result(max_abs_diff(h_ref, h_out), "max_abs_diff", spec.threshold, note);
}

PerfResult run_softmax_1d_performance(Softmax1DGpuLauncher gpu, const Softmax1DSpec &spec) {
    const Softmax1DShape shape = current_softmax_1d_shape(spec);
    std::vector<float> h_in(shape.n, 1.0f);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out(shape.n);

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_in.data(), d_out.data(), shape.n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_in.data(), d_out.data(), shape.n);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = h_in.size() * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(shape.n) * sizeof(float);
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;
    const double gelems = static_cast<double>(shape.n) / (ms / 1000.0) / 1.0e9;

    auto result = make_perf_result(ms, 0.0, "Gelem/s", gelems, spec.note, input_bytes, output_bytes, {{static_cast<size_t>(shape.n)}},
                                   {{static_cast<size_t>(shape.n)}});
    add_metric(&result, "effective_bandwidth", gbps, "GB/s");
    add_metric(&result, "elements_per_second", gelems, "Gelem/s");
    return result;
}

CorrectnessResult run_norm_row_correctness(NormRowCpuRef cpu, NormRowGpuLauncher gpu, const NormRowSpec &spec) {
    const NormRowShape requested_shape = current_norm_row_shape(spec);
    const NormRowShape shape = clamp_shape(requested_shape, norm_row_correctness_limit(spec));
    std::vector<float> h_in(static_cast<size_t>(shape.m) * shape.k);
    std::vector<float> h_ref(h_in.size());
    std::vector<float> h_out(h_in.size());
    std::string note = "reference=cpu";
    if (!shape_equals(requested_shape, shape)) {
        note = append_note(note, "requested_shape=" + format_shape(requested_shape));
        note = append_note(note, "effective_shape=" + format_shape(shape));
    }

    fill_uniform(h_in, current_seed(), -1.0f, 1.0f);
    cpu(h_in.data(), h_ref.data(), shape.m, shape.k);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out(h_out.size());

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu(d_out.data(), d_in.data(), shape.m, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.data(), h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return make_correctness_result(max_abs_diff(h_ref, h_out), "max_abs_diff", spec.threshold, note);
}

PerfResult run_norm_row_performance(NormRowGpuLauncher gpu, const NormRowSpec &spec) {
    const NormRowShape shape = current_norm_row_shape(spec);
    std::vector<float> h_in(static_cast<size_t>(shape.m) * shape.k, 1.0f);

    DeviceBuffer<float> d_in(h_in.size());
    DeviceBuffer<float> d_out(h_in.size());

    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = current_warmup(spec.warmup);
    const int iters = current_iters(spec.iters);
    for (int i = 0; i < warmup; ++i) {
        gpu(d_out.data(), d_in.data(), shape.m, shape.k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double ms = cuda_time_ms([&]() {
                          for (int i = 0; i < iters; ++i) {
                              gpu(d_out.data(), d_in.data(), shape.m, shape.k);
                          }
                          CUDA_CHECK(cudaDeviceSynchronize());
                      }) /
                      static_cast<double>(iters);

    const size_t input_bytes = h_in.size() * sizeof(float);
    const size_t output_bytes = h_in.size() * sizeof(float);
    const double gbps = static_cast<double>(input_bytes + output_bytes) / (ms / 1000.0) / 1.0e9;
    const double gelems = static_cast<double>(h_in.size()) / (ms / 1000.0) / 1.0e9;

    auto result = make_perf_result(ms, 0.0, "GB/s", gbps, spec.note, input_bytes, output_bytes,
                                   {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)}},
                                   {{static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)}});
    add_metric(&result, "elements_per_second", gelems, "Gelem/s");
    return result;
}

OpEntry make_gemm_op(const char *name, GemmCpuRef cpu, GemmGpuLauncher gpu, GemmSpec spec) {
    OpMetadata meta = build_metadata(name, "gemm",
                                     {{"M", static_cast<size_t>(spec.shape.m)}, {"N", static_cast<size_t>(spec.shape.n)},
                                      {"K", static_cast<size_t>(spec.shape.k)}},
                                     {"M", "N", "K"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_gemm_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_gemm_performance(gpu, spec); };
    return entry;
}

OpEntry make_vector_binary_op(const char *name, VectorBinaryCpuRef cpu, VectorBinaryGpuLauncher gpu, VectorBinarySpec spec) {
    OpMetadata meta =
        build_metadata(name, "vector_binary", {{"N", spec.shape.n}}, {"N"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_vector_binary_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_vector_binary_performance(gpu, spec); };
    return entry;
}

OpEntry make_transpose_op(const char *name, TransposeCpuRef cpu, TransposeGpuLauncher gpu, TransposeSpec spec) {
    OpMetadata meta = build_metadata(name, "transpose",
                                     {{"M", static_cast<size_t>(spec.shape.m)}, {"N", static_cast<size_t>(spec.shape.n)}},
                                     {"M", "N"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_transpose_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_transpose_performance(gpu, spec); };
    return entry;
}

OpEntry make_reduce_row_op(const char *name, ReduceRowCpuRef cpu, ReduceRowGpuLauncher gpu, ReduceRowSpec spec) {
    OpMetadata meta = build_metadata(name, "reduce_row",
                                     {{"M", static_cast<size_t>(spec.shape.m)}, {"K", static_cast<size_t>(spec.shape.k)}},
                                     {"M", "K"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_reduce_row_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_reduce_row_performance(gpu, spec); };
    return entry;
}

OpEntry make_topk_row_op(const char *name, TopkRowCpuRef cpu, TopkRowGpuLauncher gpu, TopkRowSpec spec) {
    OpMetadata meta = build_metadata(name, "topk_row",
                                     {{"M", static_cast<size_t>(spec.shape.m)}, {"N", static_cast<size_t>(spec.shape.n)},
                                      {"K", static_cast<size_t>(spec.shape.k)}},
                                     {"M", "N"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_topk_row_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_topk_row_performance(gpu, spec); };
    return entry;
}

OpEntry make_topk_1d_op(const char *name, Topk1DCpuRef cpu, Topk1DGpuLauncher gpu, Topk1DSpec spec) {
    OpMetadata meta = build_metadata(name, "topk_1d",
                                     {{"N", static_cast<size_t>(spec.shape.n)}, {"K", static_cast<size_t>(spec.shape.k)}},
                                     {"N"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_topk_1d_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_topk_1d_performance(gpu, spec); };
    return entry;
}

OpEntry make_softmax_1d_op(const char *name, Softmax1DCpuRef cpu, Softmax1DGpuLauncher gpu, Softmax1DSpec spec) {
    OpMetadata meta = build_metadata(name, "softmax_1d", {{"N", static_cast<size_t>(spec.shape.n)}}, {"N"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_softmax_1d_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_softmax_1d_performance(gpu, spec); };
    return entry;
}

OpEntry make_norm_row_op(const char *name, NormRowCpuRef cpu, NormRowGpuLauncher gpu, NormRowSpec spec) {
    OpMetadata meta = build_metadata(name, "norm_row",
                                     {{"M", static_cast<size_t>(spec.shape.m)}, {"K", static_cast<size_t>(spec.shape.k)}},
                                     {"M", "K"}, spec.baselines, spec.description);
    OpEntry entry;
    entry.name = meta.name;
    entry.meta = meta;
    entry.correctness = [cpu, gpu, spec]() { return run_norm_row_correctness(cpu, gpu, spec); };
    entry.performance = [gpu, spec]() { return run_norm_row_performance(gpu, spec); };
    return entry;
}

} // namespace opfw
