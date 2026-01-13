#include "ops/helpers/ActivationKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace OwnTensor {
namespace cuda {

// Constants for GELU computation
__device__ constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
__device__ constexpr float GELU_COEF = 0.044715f;

// Fast tanh approximation (faster than std::tanh on GPU)
__device__ __forceinline__ float fast_tanh(float x) {
    // Use CUDA intrinsic for tanh
    return tanhf(x);
}

// =============================================================================
// FUSED GELU KERNEL
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =============================================================================

__global__ void fused_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = fast_tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_inner);
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void fused_gelu_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time
    int64_t numel4 = numel / 4;
    for (int64_t i = idx; i < numel4; i += stride) {
        float4 x_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = (&x_vec.x)[j];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
            float tanh_inner = fast_tanh(inner);
            (&out_vec.x)[j] = 0.5f * x * (1.0f + tanh_inner);
        }
        
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }
    
    // Handle remaining elements
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = fast_tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_inner);
    }
}

void fused_gelu_cuda(
    const float* input,
    float* output,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    
    // Use vectorized kernel for large tensors where alignment is likely good
    if (numel >= 1024 && numel % 4 == 0) {
        int blocks4 = std::min((numel / 4 + threads - 1) / threads, (int64_t)65535);
        fused_gelu_kernel_vectorized<<<blocks4, threads>>>(input, output, numel);
    } else {
        fused_gelu_kernel<<<blocks, threads>>>(input, output, numel);
    }
}

// =============================================================================
// FUSED GELU BACKWARD KERNEL
// gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
// where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// and du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
// =============================================================================

__global__ void fused_gelu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float grad = grad_output[i];
        
        float x2 = x * x;
        float x3 = x2 * x;
        
        float u = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);
        
        float tanh_u = fast_tanh(u);
        float sech2_u = 1.0f - tanh_u * tanh_u;  // sech^2(u) = 1 - tanh^2(u)
        
        // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
        float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
        
        grad_input[i] = grad * gelu_grad;
    }
}

void fused_gelu_backward_cuda(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);
    fused_gelu_backward_kernel<<<blocks, threads>>>(grad_output, input, grad_input, numel);
}

// =============================================================================
// FUSED BIAS + GELU KERNEL
// output = gelu(input + bias)
// =============================================================================

__global__ void fused_bias_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t hidden_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * hidden_dim;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < total; i += stride) {
        int64_t bias_idx = i % hidden_dim;
        float x = input[i] + bias[bias_idx];
        
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_inner = fast_tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_inner);
    }
}

void fused_bias_gelu_cuda(
    const float* input,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t hidden_dim
) {
    int threads = 256;
    int64_t total = batch_size * hidden_dim;
    int blocks = std::min((total + threads - 1) / threads, (int64_t)65535);
    fused_bias_gelu_kernel<<<blocks, threads>>>(input, bias, output, batch_size, hidden_dim);
}

} // namespace cuda
} // namespace OwnTensor
