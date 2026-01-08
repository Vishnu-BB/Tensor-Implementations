#pragma once

#include "core/Tensor.h"
#include "autograd/Functions.h"
#include "autograd/Node.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware operations that automatically build computational graph.
 * 
 * These functions:
 * 1. Perform the forward computation
 * 2. If inputs require_grad, attach grad_fn to result
 * 3. Set up graph edges for backward pass
 */

/**
 * @brief Autograd-aware addition
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware multiplication
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware matrix multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware ReLU
 */
Tensor relu(const Tensor& x);

/**
 * @brief Autograd-aware sum
 */
Tensor sum(const Tensor& x);

/**
 * @brief Autograd-aware mean
 */
Tensor mean(const Tensor& x);

} // namespace autograd
} // namespace OwnTensor
