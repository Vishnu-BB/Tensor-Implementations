#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware matrix multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware linear transformation: x @ W + b
 * Fuse matmul and bias add for better performance.
 */
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);



} // namespace autograd
} // namespace OwnTensor