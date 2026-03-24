#include <algorithm>
#include <cfloat>

#include "op_runtime.hpp"

// 辅助函数
__device__ __forceinline__ void get_max_sum(float& sum_self, float& max_self){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        float sum_other = __shfl_down_sync(0xffffffff, sum_self, offset);
        float max_other = __shfl_down_sync(0xffffffff, max_self, offset);
        float max_temp = fmaxf(max_other, max_self);
        
        // Online Softmax 修正公式
        sum_self = sum_self * __expf(max_self - max_temp) + sum_other * __expf(max_other - max_temp);
        max_self = max_temp;
    }
}

// 存储所有线程块的max和sum
__device__ float g_block_max[1024]; // 足够容纳所有 SM 的输出
__device__ float g_block_sum[1024];

// 所有线程块内部进行online softmax 得到局部sum和max
template<int threadsPerBlock>
__global__ void softmax_pass1_kernel(const float* input, int N) {
    // 共享内存：用来存放每个 Warp 规约后的结果
    __shared__ float smem_data[(threadsPerBlock / 32) * 2];
    
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;

    float max_self = -FLT_MAX;
    float sum_self = 0.0f;

    // 网格跨步循环：每个线程扫一遍自己负责的所有数据
    for (int i = id; i < N; i += stride) {
        float val = input[i];
        float max_prev = max_self;
        max_self = fmaxf(max_prev, val);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(val - max_self);
    }

    // warp内部规约
    get_max_sum(sum_self, max_self);

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = threadsPerBlock / 32;

    if (lane_id == 0) {
        smem_data[warp_id] = sum_self;
        smem_data[warp_id + num_warps] = max_self;
    }
    __syncthreads();

    // 二次规约求出block的局部sum和max
    if (warp_id == 0) {
        float b_sum = (tid < num_warps) ? smem_data[tid] : 0.0f;
        float b_max = (tid < num_warps) ? smem_data[tid + num_warps] : -FLT_MAX;
        get_max_sum(b_sum, b_max);
        
        // 数据写入到全局内存
        if (tid == 0) {
            g_block_max[blockIdx.x] = b_max;
            g_block_sum[blockIdx.x] = b_sum;
        }
    }
}

// 汇总所有线程块的结果 得到最终输出
template<int threadsPerBlock>
__global__ void softmax_pass2_kernel(const float* input, float* output, int N, int num_blocks) {
    __shared__ float final_G_max, final_G_sum;

    // 由第一个 Warp 负责汇总那 blocksPerGrid 个块的局部情报
    if (threadIdx.x < 32) {
        float b_max = -FLT_MAX;
        float b_sum = 0.0f;
        
        // 获取所有线程块的数据
        for(int i = threadIdx.x; i < num_blocks; i += 32) {
            float m = g_block_max[i];
            float s = g_block_sum[i];
            float m_prev = b_max;
            b_max = fmaxf(m_prev, m);
            b_sum = b_sum * __expf(m_prev - b_max) + s * __expf(m - b_max);
        }
        
        // Warp 内再次规约对齐
        get_max_sum(b_sum, b_max);
        if (threadIdx.x == 0) {
            final_G_max = b_max;
            final_G_sum = b_sum;
        }
    }
    __syncthreads(); // 此时块内所有线程都拿到了全局真值

    // 输出结果
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < N; i += stride) {
        // 利用全局 Max 和 Sum 归一化
        output[i] = __expf(input[i] - final_G_max) / final_G_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 512;
    
    // 获取设备 SM 数量
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    
    // 我们固定启动一定数量的 Block，确保 Occupancy 足够
    // 同时也让 g_block_max/sum 数组大小可控
    int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, sm_count * 4);

    // 获取所有线程块的局部max和sum
    softmax_pass1_kernel<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(input, N);
    CUDA_CHECK(cudaGetLastError());
    
    // 基于kernel1的结果进一步处理
    softmax_pass2_kernel<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(input, output, N, blocksPerGrid);
    CUDA_CHECK(cudaGetLastError());
    
}

LC_REGISTER_SOFTMAX_1D_OP("softmax", opfw::cpu_softmax_1d_f32, solve);
