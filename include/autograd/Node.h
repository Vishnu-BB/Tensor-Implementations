#pragma once

#include <memory>
#include <vector>
#include <functional>

namespace OwnTensor {

// Forward declarations
class Node;
class Tensor;

/**
 * @brief Edge in the computational graph connecting nodes.
 * 
 * An edge represents a connection from one node to another in the backward
 * computational graph. It stores which function to call next and which input
 * slot of that function this edge corresponds to.
 */
struct Edge {
    std::shared_ptr<Node> function;  ///< Next function in backward graph
    uint32_t input_nr;                ///< Which input of the function (for multi-input ops)

    Edge() : function(nullptr), input_nr(0) {}
    
    Edge(std::shared_ptr<Node> function_, uint32_t input_nr_)
        : function(std::move(function_)), input_nr(input_nr_) {}

    /// Check if this edge is valid (has a function)
    bool is_valid() const {
        return function != nullptr;
    }
};

/**
 * @brief Base class for gradient functions (operations in computational graph).
 * 
 * Node represents an operation in the computational graph. During the backward
 * pass, nodes receive gradients and compute gradients for their inputs.
 * 
 * Each node knows:
 * - How to compute gradients (via apply())
 * - Which nodes come next in the graph (next_edges_)
 * - How many times it's been used (for multi-output ops)
 * 
 * NOTE: This is a simplified Node for cgadimpl/tensor.
 * Full backward() implementation will be added later.
 */
class Node : public std::enable_shared_from_this<Node> {
public:
    Node() = default;
    
    explicit Node(uint32_t num_inputs) {
        next_edges_.resize(num_inputs);
    }

    virtual ~Node() = default;

    /**
     * @brief Apply the gradient function (backward pass).
     * 
     * @param grads Gradients with respect to outputs of this function
     * @return Gradients with respect to inputs of this function
     * 
     * Example: For y = f(x1, x2), given dy, compute dx1 and dx2.
     */
    virtual std::vector<Tensor> apply(std::vector<Tensor>&& grads) = 0;

    /**
     * @brief Get the edges to the next functions in the graph.
     */
    const std::vector<Edge>& next_edges() const {
        return next_edges_;
    }

    /**
     * @brief Set an edge for a specific input.
     */
    void set_next_edge(uint32_t index, Edge edge) {
        if (index >= next_edges_.size()) {
            next_edges_.resize(index + 1);
        }
        next_edges_[index] = std::move(edge);
    }

    /**
     * @brief Add an edge to the list.
     */
    void add_next_edge(Edge edge) {
        next_edges_.push_back(std::move(edge));
    }

    /**
     * @brief Get number of inputs to this function.
     */
    size_t num_inputs() const {
        return next_edges_.size();
    }

    /**
     * @brief Clear all next edges (used for graph cleanup).
     */
    void clear_edges() {
        next_edges_.clear();
    }

protected:
    /// Edges to the next functions in the backward graph
    /// next_edges_[i] corresponds to the i-th input of this function
    std::vector<Edge> next_edges_;
};

// Helper to create edges more easily
inline Edge make_edge(std::shared_ptr<Node> node, uint32_t input_nr = 0) {
    return Edge(std::move(node), input_nr);
}

} // namespace OwnTensor
