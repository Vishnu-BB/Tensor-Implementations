#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware matrix multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware embedding lookup
 */
Tensor embedding(const Tensor& indices, const Tensor& weight, int padding_idx = -1);

} // namespace autograd
} // namespace OwnTensor