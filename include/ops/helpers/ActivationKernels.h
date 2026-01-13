#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Fused GELU activation kernel
 * 
 * Computes GELU in a single kernel launch instead of 6+ separate operations:
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void fused_gelu_cuda(
    const float* input,
    float* output,
    int64_t numel
);

/**
 * @brief Fused GELU backward kernel
 * 
 * Computes gradient in a single kernel launch.
 */
void fused_gelu_backward_cuda(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int64_t numel
);

/**
 * @brief Fused bias + GELU kernel
 * 
 * Computes output = gelu(input + bias) in a single kernel.
 * bias is broadcast along the last dimension.
 */
void fused_bias_gelu_cuda(
    const float* input,
    const float* bias,
    float* output,
    int64_t batch_size,  // Total elements / hidden_dim
    int64_t hidden_dim   // Size of bias vector
);

} // namespace cuda
} // namespace OwnTensor
