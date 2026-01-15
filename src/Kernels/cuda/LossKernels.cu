#include "ops/helpers/LossKernels.h"
#include "dtype/Types.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

namespace OwnTensor {
namespace cuda {

template<typename T, typename T_idx>
__global__ void sparse_ce_backward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* grad,
    int64_t batch_size,
    int64_t vocab_size,
    T scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // Use float for internal accumulation to ensure stability and compatibility
    float max_v = -1e38f; 
    for (int64_t j = 0; j < vocab_size; ++j) {
        float val = static_cast<float>(logits[i * vocab_size + j]);
        if (val > max_v) max_v = val;
    }

    float sum_exp = 0.0f;
    for (int64_t j = 0; j < vocab_size; ++j) {
        sum_exp += expf(static_cast<float>(logits[i * vocab_size + j]) - max_v);
    }

    int64_t target_idx = static_cast<int64_t>(targets[i]);
    for (int64_t j = 0; j < vocab_size; ++j) {
        float p = expf(static_cast<float>(logits[i * vocab_size + j]) - max_v) / sum_exp;
        float is_target = (j == target_idx) ? 1.0f : 0.0f;
        // scale is of type T, we convert to float for math
        float f_scale = static_cast<float>(scale);
        grad[i * vocab_size + j] = static_cast<T>((p - is_target) * f_scale);
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

} // namespace cuda
} // namespace OwnTensor
