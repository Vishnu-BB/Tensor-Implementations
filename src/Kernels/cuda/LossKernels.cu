#include "ops/helpers/LossKernels.h"
#include "dtype/Types.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

namespace OwnTensor {
namespace cuda {

// ============================================================================
// Forward Pass CUDA Kernels
// ============================================================================

/**
 * @brief Compute sparse cross entropy loss for each sample (forward pass)
 * 
 * Uses numerically stable log-softmax:
 * loss[i] = log(sum(exp(logits[i] - max(logits[i])))) + max(logits[i]) - logits[i, target[i]]
 */
template<typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* losses,
    int64_t batch_size,
    int64_t vocab_size
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // Find max for numerical stability (prevent overflow in exp)
    float max_v = -1e38f;
    for (int64_t j = 0; j < vocab_size; ++j) {
        float val = static_cast<float>(logits[i * vocab_size + j]);
        if (val > max_v) max_v = val;
    }

    // Compute sum of exp(logits - max)
    float sum_exp = 0.0f;
    for (int64_t j = 0; j < vocab_size; ++j) {
        sum_exp += expf(static_cast<float>(logits[i * vocab_size + j]) - max_v);
    }

    // Get target index
    int64_t target_idx = static_cast<int64_t>(targets[i]);
    
    // Compute log_softmax[target] = logits[target] - max - log(sum_exp)
    // Loss = -log_softmax[target] = log(sum_exp) + max - logits[target]
    float log_sum_exp = max_v + logf(sum_exp);
    float target_logit = static_cast<float>(logits[i * vocab_size + target_idx]);
    float loss = log_sum_exp - target_logit;
    
    losses[i] = static_cast<T>(loss);
}

/**
 * @brief Parallel reduction to sum all losses
 * Uses shared memory for efficient reduction
 */
template<typename T>
__global__ void sum_reduction_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? static_cast<float>(input[i]) : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = static_cast<T>(sdata[0]);
    }
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    if (batch_size == 0) {
        // Set loss to 0
        cudaMemsetAsync(loss_output, 0, sizeof(T), stream);
        return;
    }
    
    // Allocate temporary buffer for per-sample losses
    T* d_losses;
    cudaMalloc(&d_losses, batch_size * sizeof(T));
    
    // Kernel 1: Compute loss for each sample
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    sparse_ce_forward_kernel_typed<T, T_idx><<<blocks, threads, 0, stream>>>(
        logits, targets, d_losses, batch_size, vocab_size
    );
    
    // Kernel 2: Reduce to sum all losses
    // For simplicity, use a two-pass reduction
    if (batch_size <= 1024) {
        // Single block reduction
        int reduce_threads = 256;
        int reduce_blocks = 1;
        sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, loss_output, batch_size
        );
    } else {
        // Two-pass reduction for large batches
        int reduce_threads = 256;
        int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;
        
        T* d_partial;
        cudaMalloc(&d_partial, reduce_blocks * sizeof(T));
        
        // First reduction
        sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, d_partial, batch_size
        );
        
        // Second reduction
        sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_partial, loss_output, reduce_blocks
        );
        
        cudaFree(d_partial);
    }
    
    cudaFree(d_losses);
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    sparse_cross_entropy_forward_cuda_impl<T, T_idx>(
        logits, targets, loss_output, batch_size, vocab_size, stream
    );
}

// ============================================================================
// Backward Pass CUDA Kernels (existing)
// ============================================================================

/**
 * @brief CUDA kernel for sparse cross entropy backward pass
 * 
 * Gradient formula (derived from calculus):
 *   For class j:
 *     grad[j] = softmax(logits)[j] * scale           if j != target
 *     grad[j] = (softmax(logits)[j] - 1.0) * scale   if j == target
 * 
 * This is equivalent to: grad = (softmax - target_indicator) * scale
 * where target_indicator equals 1 only at the target class index.
 * 
 * @param logits     Input logits [batch_size, vocab_size]
 * @param targets    Sparse target indices [batch_size] (NOT one-hot!)
 * @param grad       Output gradient [batch_size, vocab_size]
 * @param batch_size Number of samples
 * @param vocab_size Number of classes
 * @param scale      Gradient scaling factor (typically 1/batch_size)
 */
template<typename T, typename T_idx>
__global__ void sparse_ce_backward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* grad,
    int64_t batch_size,
    int64_t vocab_size,
    T scale
) {
    int64_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    // Get pointer to this sample's logits and gradients
    const T* sample_logits = logits + sample_idx * vocab_size;
    T* sample_grad = grad + sample_idx * vocab_size;
    
    // Get the target class for this sample (sparse index, not one-hot)
    int64_t target_class = static_cast<int64_t>(targets[sample_idx]);
    
    // ========================================================================
    // Step 1: Find max logit for numerical stability
    // ========================================================================
    float max_logit = -1e38f;
    for (int64_t c = 0; c < vocab_size; ++c) {
        float logit_val = static_cast<float>(sample_logits[c]);
        if (logit_val > max_logit) max_logit = logit_val;
    }
    
    // ========================================================================
    // Step 2: Compute sum of exp(logits - max) for softmax denominator
    // ========================================================================
    float sum_exp = 0.0f;
    for (int64_t c = 0; c < vocab_size; ++c) {
        sum_exp += expf(static_cast<float>(sample_logits[c]) - max_logit);
    }
    
    // ========================================================================
    // Step 3: Compute gradient for each class
    // 
    // The gradient derivation:
    //   Loss = -log(p_target) where p = softmax(logits)
    //   
    //   For non-target classes (c != target):
    //     dL/d(logit_c) = p_c
    //   
    //   For target class (c == target):
    //     dL/d(logit_c) = p_c - 1
    //
    // This can be written as: grad_c = p_c - (c == target ? 1 : 0)
    // ========================================================================
    float f_scale = static_cast<float>(scale);
    
    for (int64_t c = 0; c < vocab_size; ++c) {
        // Compute softmax probability for this class
        float prob = expf(static_cast<float>(sample_logits[c]) - max_logit) / sum_exp;
        
        // Compute gradient:
        // - For non-target classes: grad = prob * scale
        // - For target class: grad = (prob - 1) * scale
        float grad_val;
        if (c == target_class) {
            grad_val = (prob - 1.0f) * f_scale;  // Target class: subtract 1
        } else {
            grad_val = prob * f_scale;           // Other classes: just softmax
        }
        
        sample_grad[c] = static_cast<T>(grad_val);
    }
}

// Specialization/Guard for non-numeric types if needed, but static_cast<float> should work for most primitives.
// Complex types will still fail to static_cast to float.

// We will restrict the instantiations to real numeric types.

template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    T scale,
    cudaStream_t stream
) {
    if (batch_size == 0) return;
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    sparse_ce_backward_kernel_typed<T, T_idx><<<blocks, threads, 0, stream>>>(
        logits, targets, grad_logits, batch_size, vocab_size, scale
    );
}

template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    T scale,
    cudaStream_t stream
) {
    sparse_cross_entropy_backward_cuda_impl<T, T_idx>(logits, targets, grad_logits, batch_size, vocab_size, scale, stream);
}

// Explicit instantiations for supported types
#define INSTANTIATE_GIVEN_T(T) \
    template void sparse_cross_entropy_backward_cuda<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, T, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, T, cudaStream_t);

INSTANTIATE_GIVEN_T(float)
INSTANTIATE_GIVEN_T(double)
INSTANTIATE_GIVEN_T(float16_t)
INSTANTIATE_GIVEN_T(bfloat16_t)

// Explicit instantiations for forward pass
#define INSTANTIATE_FORWARD_GIVEN_T(T) \
    template void sparse_cross_entropy_forward_cuda<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, cudaStream_t);

INSTANTIATE_FORWARD_GIVEN_T(float)
INSTANTIATE_FORWARD_GIVEN_T(double)
INSTANTIATE_FORWARD_GIVEN_T(float16_t)
INSTANTIATE_FORWARD_GIVEN_T(bfloat16_t)

} // namespace cuda
} // namespace OwnTensor
