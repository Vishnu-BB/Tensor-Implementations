#include "ops/helpers/EmbeddingKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace OwnTensor {
namespace cuda {

__global__ void embedding_forward_kernel(
    const uint16_t* indices,
    const float* weight,
    float* output,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
) {
    int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    uint16_t tok = indices[n];
    float* out_row = output + n * C;

    if (tok == (uint16_t)padding_idx) {
        for (int64_t j = 0; j < C; ++j) {
            out_row[j] = 0.0f;
        }
        return;
    }

    if (tok >= V) {
        // In CUDA kernels we usually don't throw, but we can't do much here.
        // We just skip invalid tokens.
        return;
    }

    const float* w_row = weight + (size_t)tok * C;
    for (int64_t j = 0; j < C; ++j) {
        out_row[j] = w_row[j];
    }
}

__global__ void embedding_backward_kernel(
    const uint16_t* indices,
    const float* grad_output,
    float* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
) {
    int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    uint16_t tok = indices[n];
    if (tok == (uint16_t)padding_idx || tok >= V) return;

    const float* go_row = grad_output + n * C;
    float* gw_row = grad_weight + (size_t)tok * C;

    for (int64_t j = 0; j < C; ++j) {
        atomicAdd(gw_row + j, go_row[j]);
    }
}

void embedding_forward_cuda(
    const uint16_t* indices,
    const float* weight,
    float* output,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    embedding_forward_kernel<<<blocks, threads>>>(indices, weight, output, N, C, V, padding_idx);
}

void embedding_backward_cuda(
    const uint16_t* indices,
    const float* grad_output,
    float* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    embedding_backward_kernel<<<blocks, threads>>>(indices, grad_output, grad_weight, N, C, V, padding_idx);
}

} // namespace cuda
} // namespace OwnTensor