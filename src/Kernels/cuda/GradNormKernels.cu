#include "ops/helpers/GradNormKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// FUSED GRADIENT NORM KERNEL
// Computes sum of squares and atomically adds to accumulator
// =============================================================================

__global__ void grad_norm_squared_kernel(
    const float* __restrict__ grad,
    float* __restrict__ norm_sq_accumulator,
    int64_t numel
) {
    // Use shared memory for block-level reduction
    extern __shared__ float sdata[];
    
    int64_t tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Each thread computes sum of squares for its elements
    float thread_sum = 0.0f;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = grad[i];
        thread_sum += val * val;
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // First thread atomically adds block result to global accumulator
    if (tid == 0) {
        atomicAdd(norm_sq_accumulator, sdata[0]);
    }
}

void grad_norm_squared_cuda(
    const float* grad,
    float* norm_sq_accumulator,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)1024);
    size_t smem_size = threads * sizeof(float);
    
    grad_norm_squared_kernel<<<blocks, threads, smem_size>>>(
        grad, norm_sq_accumulator, numel
    );
}

// =============================================================================
// SCALE GRADIENTS KERNEL
// =============================================================================

__global__ void scale_gradients_kernel(
    float* __restrict__ grad,
    float scale,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Vectorized path using float4
    int64_t numel4 = numel / 4;
    for (int64_t i = idx; i < numel4; i += stride) {
        float4* ptr = reinterpret_cast<float4*>(grad) + i;
        float4 val = *ptr;
        val.x *= scale;
        val.y *= scale;
        val.z *= scale;
        val.w *= scale;
        *ptr = val;
    }
    
    // Handle remaining elements
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        grad[i] *= scale;
    }
}

void scale_gradients_cuda(
    float* grad,
    float scale,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel / 4 + threads - 1) / threads, (int64_t)65535);
    if (blocks == 0) blocks = 1;
    
    scale_gradients_kernel<<<blocks, threads>>>(grad, scale, numel);
}

} // namespace cuda
} // namespace OwnTensor
