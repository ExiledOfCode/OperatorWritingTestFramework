// topk_desc.cu
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <math_constants.h> // CUDART_INF_F

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(_e));                                     \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// -------------------------
// warp helpers
// -------------------------
__device__ __forceinline__ int warpInclusiveScan(int x, unsigned mask = 0xffffffffu) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int y = __shfl_up_sync(mask, x, offset);
        if ((threadIdx.x & 31) >= offset) x += y;
    }
    return x;
}

__device__ __forceinline__ int warpReduceSum(int x, unsigned mask = 0xffffffffu) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset);
    }
    return x; // valid only in lane0
}

// -------------------------
// Radix sort (ascending) infra
// -------------------------
static constexpr int nThreads = 1024;
static constexpr int TILE     = 1024;
static constexpr int R        = 2;
static constexpr int RADIX    = 1 << R;

__global__ void reduce_kernel(const int* input, int* output, int* block_sum, int N) {
    __align__(16) __shared__ int warp_sum[32 * RADIX];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    int id = blockIdx.x * TILE + threadIdx.x;

    __align__(16) int v[RADIX] = {0};
    if (id < N) {
        // RADIX=4 here, use int4 load/store
        reinterpret_cast<int4*>(&v[0])[0] = reinterpret_cast<const int4*>(&input[id * RADIX])[0];
    }

    #pragma unroll
    for (int i = 0; i < RADIX; i++) v[i] = warpInclusiveScan(v[i]);

    if (lane_id == 31) {
        #pragma unroll
        for (int i = 0; i < RADIX; i++) warp_sum[warp_id * RADIX + i] = v[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < RADIX; i++) {
            int prefix = warp_sum[lane_id * RADIX + i];
            prefix = warpInclusiveScan(prefix);
            warp_sum[lane_id * RADIX + i] = prefix;
        }
    }
    __syncthreads();

    // add previous-warp totals
    {
        int4 tmp  = reinterpret_cast<int4*>(&v[0])[0];
        int4 prev = make_int4(0, 0, 0, 0);
        if (warp_id != 0) {
            prev = reinterpret_cast<int4*>(&warp_sum[(warp_id - 1) * RADIX])[0];
        }
        tmp.x += prev.x; tmp.y += prev.y; tmp.z += prev.z; tmp.w += prev.w;

        if (id < N) {
            reinterpret_cast<int4*>(&output[id * RADIX])[0] = tmp;
        }
        if (threadIdx.x == nThreads - 1) {
            reinterpret_cast<int4*>(&block_sum[blockIdx.x * RADIX])[0] = tmp;
        }
    }
}

__global__ void scatter_kernel(int* output, int* block_prefix, int N) {
    int id = blockIdx.x * TILE + threadIdx.x;
    if (id < N) {
        int4 tmp = reinterpret_cast<int4*>(&output[id * RADIX])[0];
        int4 add = make_int4(0, 0, 0, 0);
        if (blockIdx.x != 0) {
            add = reinterpret_cast<int4*>(&block_prefix[(blockIdx.x - 1) * RADIX])[0];
        }
        tmp.x += add.x; tmp.y += add.y; tmp.z += add.z; tmp.w += add.w;
        reinterpret_cast<int4*>(&output[id * RADIX])[0] = tmp;
    }
}

static inline void build_levels(int N, std::vector<int>& n_list, std::vector<int>& nb_list) {
    n_list.clear(); nb_list.clear();
    int curN = N;
    while (true) {
        int nb = (curN + TILE - 1) / TILE;
        n_list.push_back(curN);
        nb_list.push_back(nb);
        if (nb == 1) break;
        curN = nb;
    }
}

void reduce_prefix_sum(const int* input, int* output, int N) {
    std::vector<int> n_list, nb_list;
    build_levels(N, n_list, nb_list);
    const int L = (int)nb_list.size();

    size_t sum_elems = 0;
    size_t out_elems = 0;
    for (int l = 0; l < L; l++) sum_elems += (size_t)nb_list[l] * RADIX;
    for (int l = 1; l < L; l++) out_elems += (size_t)nb_list[l - 1] * RADIX;

    int* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, (sum_elems + out_elems) * sizeof(int)));

    int* sums_base = scratch;
    int* outs_base = scratch + sum_elems;

    std::vector<int*> sums(L);
    std::vector<int*> outs(L, nullptr);
    outs[0] = output;

    {
        size_t off = 0;
        for (int l = 0; l < L; l++) {
            sums[l] = sums_base + off;
            off += (size_t)nb_list[l] * RADIX;
        }
        off = 0;
        for (int l = 1; l < L; l++) {
            outs[l] = outs_base + off;
            off += (size_t)nb_list[l - 1] * RADIX;
        }
    }

    // Up-sweep
    const int* cur_in = input;
    for (int l = 0; l < L; l++) {
        int curN = n_list[l];
        int nb   = nb_list[l];
        int* cur_out = outs[l];
        int* cur_sum = sums[l];
        reduce_kernel<<<nb, nThreads>>>(cur_in, cur_out, cur_sum, curN);
        CUDA_CHECK(cudaGetLastError());
        cur_in = cur_sum;
    }

    // Down-sweep
    for (int l = L - 2; l >= 0; --l) {
        int nb = nb_list[l];
        int n  = n_list[l];
        scatter_kernel<<<nb, nThreads>>>(outs[l], outs[l + 1], n);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(scratch));
}

__device__ __forceinline__ uint32_t float_to_radix_key(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t mask = (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask; // ascending float order
}

__global__ void radix_scan(const float* input, int d, int* block_sum, int N) {
    __shared__ int warp_sum[32 * RADIX];
    int id = blockIdx.x * TILE + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int key = -1;
    if (id < N) {
        uint32_t u = float_to_radix_key(input[id]);
        key = (u >> d) & (RADIX - 1);
    }

    #pragma unroll
    for (int i = 0; i < RADIX; i++) {
        int val = (key == i);
        val = warpReduceSum(val);
        if (lane_id == 0) warp_sum[warp_id * RADIX + i] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < RADIX; i++) {
            int val = warpReduceSum(warp_sum[lane_id * RADIX + i]);
            if (lane_id == 0) block_sum[blockIdx.x * RADIX + i] = val;
        }
    }
}

__global__ void radix_scatter(const float* input, float* output, int d, const int* block_prefix, int N) {
    __shared__ int warp_sum[32 * RADIX];
    int id = blockIdx.x * TILE + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int key = -1;
    float x = 0.0f;
    if (id < N) {
        x = input[id];
        uint32_t u = float_to_radix_key(x);
        key = (u >> d) & (RADIX - 1);
    }

    int warp_prev = 0;
    #pragma unroll
    for (int i = 0; i < RADIX; i++) {
        int val = (key == i);
        val = warpInclusiveScan(val);
        if (i == key) warp_prev = val;
        if (lane_id == 31) warp_sum[warp_id * RADIX + i] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < RADIX; i++) {
            int prefix = warp_sum[lane_id * RADIX + i];
            prefix = warpInclusiveScan(prefix);
            warp_sum[lane_id * RADIX + i] = prefix;
        }
    }
    __syncthreads();

    if (id < N) {
        int sum = warp_prev;

        // add previous warps in this block for this bucket
        if (warp_id != 0) sum += warp_sum[(warp_id - 1) * RADIX + key];

        // add previous blocks prefix for this bucket
        if (blockIdx.x != 0) sum += block_prefix[(blockIdx.x - 1) * RADIX + key];

        // add totals of buckets < key
        #pragma unroll
        for (int i = 0; i < RADIX; i++) {
            if (i < key) sum += block_prefix[(gridDim.x - 1) * RADIX + i];
        }

        output[sum - 1] = x;
    }
}

// Gather top-k largest in DESC order from an ASC-sorted array
__global__ void gather_topk_desc_from_sorted_asc(const float* sorted_asc, float* out_desc, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        out_desc[i] = sorted_asc[N - 1 - i];
    }
}

void radix_solve(const float* input, float* output, int N, int K) {
    int block_cnt = (N + TILE - 1) / TILE;

    int* block_sum = nullptr;
    int* block_prefix = nullptr;
    float* temp0 = nullptr;
    float* temp1 = nullptr;

    CUDA_CHECK(cudaMalloc(&block_sum,    block_cnt * RADIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&block_prefix, block_cnt * RADIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp0, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp1, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(temp0, input, N * sizeof(float), cudaMemcpyDeviceToDevice));

    float* buf[2] = { temp0, temp1 };
    int flip = 0;

    for (int d = 0; d < 32; d += R) {
        radix_scan<<<block_cnt, nThreads>>>(buf[flip], d, block_sum, N);
        CUDA_CHECK(cudaGetLastError());

        reduce_prefix_sum(block_sum, block_prefix, block_cnt);

        radix_scatter<<<block_cnt, nThreads>>>(buf[flip], buf[flip ^ 1], d, block_prefix, N);
        CUDA_CHECK(cudaGetLastError());

        flip ^= 1;
    }

    // buf[flip] is fully sorted ASC. Gather largest K in DESC order.
    int threads = 256;
    int blocks  = (K + threads - 1) / threads;
    gather_topk_desc_from_sorted_asc<<<blocks, threads>>>(buf[flip], output, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(block_sum));
    CUDA_CHECK(cudaFree(block_prefix));
    CUDA_CHECK(cudaFree(temp0));
    CUDA_CHECK(cudaFree(temp1));
}

// -------------------------
// Bitonic top-k (DESC) per block
// -------------------------
template <bool kAscending>
__device__ __forceinline__ void bitonic_compare_swap(float &a, float &b) {
    if constexpr (kAscending) {
        if (a > b) { float t = a; a = b; b = t; }
    } else {
        if (a < b) { float t = a; a = b; b = t; }
    }
}

// Sort DESC within each block and output first K (largest K, descending)
__global__ void topk_kernel_desc(const float* a, float* b, int N, int K) {
    __shared__ float ks[nThreads];
    int tid = threadIdx.x;
    int id  = blockIdx.x * nThreads + tid;

    // pad with -inf so padding sinks to the end in DESC sort
    ks[tid] = (id < N) ? a[id] : -CUDART_INF_F;
    __syncthreads();

    // Bitonic network: flip the stage direction to get overall DESC
    for (int stage = 2; stage <= nThreads; stage <<= 1) {
        for (int j = stage >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool dir_asc = ((tid & stage) == 0);
                dir_asc = !dir_asc; // invert => overall DESC

                if (dir_asc) bitonic_compare_swap<true>(ks[tid], ks[ixj]);
                else         bitonic_compare_swap<false>(ks[tid], ks[ixj]);
            }
            __syncthreads();
        }
    }

    if (tid < K) {
        // already descending: ks[0] >= ks[1] >= ...
        b[blockIdx.x * K + tid] = ks[tid];
    }
}

void inner_solve(const float* input, float* output, int N, int K) {
    if (K > nThreads) {
        fprintf(stderr, "inner_solve requires K <= nThreads (%d)\n", nThreads);
        exit(1);
    }

    int block_num = (N + nThreads - 1) / nThreads;
    float* tmp[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&tmp[0], block_num * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp[1], block_num * K * sizeof(float)));

    int hold = N;
    int cnt = 0;
    while (cnt == 0 || hold > K) {
        int bc = (hold + nThreads - 1) / nThreads;
        int flag = cnt % 2;

        if (cnt == 0) {
            topk_kernel_desc<<<bc, nThreads>>>(input, tmp[1], hold, K);
        } else {
            topk_kernel_desc<<<bc, nThreads>>>(tmp[flag], tmp[flag ^ 1], hold, K);
        }
        CUDA_CHECK(cudaGetLastError());

        hold = bc * K; // candidate count after this pass
        cnt++;
    }

    int flag = cnt % 2;
    CUDA_CHECK(cudaMemcpy(output, tmp[flag], K * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(tmp[0]));
    CUDA_CHECK(cudaFree(tmp[1]));
}

// -------------------------
// Entry
// -------------------------
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (k <= 0) return;
    if (k > N) k = N;

    // Choose path: radix for large k, bitonic topk for small k
    if (k > 100) {
        radix_solve(input, output, N, k);      // output descending
    } else {
        inner_solve(input, output, N, k);      // output descending
    }
}
