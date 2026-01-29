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
 * @brief Execution mode for autograd backward pass.
 * 
 * SEQUENTIAL: Single-threaded, dependency-based execution (default)
 * PARALLEL: Multi-threaded execution using thread pool
 */
enum class ExecutionMode {
    SEQUENTIAL,  // Default: deterministic, dependency-based sequential execution
    PARALLEL     // Thread-pool based parallel execution with async dependencies
};

/**
 * @brief Get current execution mode.
 * 
 * @return Current ExecutionMode (global setting)
 */
ExecutionMode get_execution_mode();

/**
 * @brief Set execution mode for backward passes.
 * 
 * @param mode ExecutionMode to use (SEQUENTIAL or PARALLEL)
 * 
 * Sequential mode (default):
 * - Single-threaded execution following topological order
 * - Dependency counter decrements as nodes complete
 * - Deterministic, easier to debug
 * - May be slower on multi-core systems
 * 
 * Parallel mode:
 * - Multi-threaded execution using thread pool
 * - Asynchronous dependency resolution with mutex/condition variables
 * - Faster on multi-core systems
 * - Non-deterministic execution order
 */
void set_execution_mode(ExecutionMode mode);

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
