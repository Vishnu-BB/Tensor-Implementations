#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace OwnTensor {

/**
 * @brief Autograd engine for backward pass execution.
 * 
 * Responsible for:
 * - Topological sorting of computational graph
 * - Gradient propagation through the graph
 * - Handling multi-output operations
 */
namespace autograd {

/**
 * @brief Perform topological sort on computational graph.
 * 
 * @param root Root tensor to start from
 * @return Nodes in topological order (leaves first, root last)
 */
std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root);

/**
 * @brief Execute backward pass from root tensor.
 * 
 * @param root Tensor to compute gradients for
 * @param grad_output Initial gradient (default: ones like root)
 */
void backward(const Tensor& root, const Tensor* grad_output = nullptr);

} // namespace autograd
} // namespace OwnTensor
