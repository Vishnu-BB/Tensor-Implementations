#include "autograd/Engine.h"
#include "autograd/Functions.h"
#include "core/AutogradMeta.h"
#include "core/TensorImpl.h"
#include "ops/TensorOps.h"
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include<iostream>
namespace OwnTensor {
namespace autograd {

void build_topo_recursive(Node* node, std::unordered_set<Node*>& visited, std::vector<std::shared_ptr<Node>>& result) {
    if (visited.count(node)) return;
    visited.insert(node);
    
    // Visit inputs (next edges)
    for (const auto& edge : node->next_edges()) {
        if (edge.is_valid()) {
            build_topo_recursive(edge.function.get(), visited, result);
        }
    }
    
    // Post-order add to result (leaves first) -> Wait, we want topological order for Backward.
    // Backward needs: if Node A inputs to Node B, process B then A? 
    // No. In Backward: Root (Loss) -> ... -> Leaf.
    // If A takes B as input in forward. Forward: B -> A.
    // Backward: A -> B.
    // So we need to process A before B.
    // This means Topological Sort of the Backward Graph.
    // Backward Graph edges: A -> B.
    // Standard DFS Post-Order gives Reverse Topological Sort.
    // So if we push_back in post-order, we satisfy dependency if we read backwards?
    // Let's trace:
    // Visit A. Calls B. Visit B. B finishes. Push B. A finishes. Push A.
    // Result: [B, A].
    // Dependency: A -> B (A passes grad to B).
    // So we need to process A, then B.
    // So we need [A, B].
    // This is Reverse Post-Order.
    // So we just reverse the result vector? Or push_front (slow).
    // Or just read vector in reverse?
    // `Engine` iterates `nodes` forward.
    // So `nodes` should be [A, B].
    // So we need Reverse result of Post-Order DFS.
}

std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root) {
    std::vector<std::shared_ptr<Node>> result;
    std::unordered_set<Node*> visited;
    
    auto root_fn = root.grad_fn();
    if (!root_fn) return result;
    
    // DFS
    // We need to capture shared_ptrs to keep nodes alive?
    // The graph holds shared_ptrs.
    // DFS traverses raw pointers but we need to store shared_ptr in result.
    // We can't get shared_ptr from raw pointer easily without `shared_from_this`.
    // Node inherits `enable_shared_from_this`.
    
    // Helper lambda to handle shared_ptr
    // std::function must handle recursion
    std::vector<std::shared_ptr<Node>> stack;
    // Iterative DFS to avoid recursion depth issues? (Depth ~100 is fine).
    // But how to get shared_ptr from Node* in 'visited'?
    // Just pass shared_ptr to recursive function.
    
    // Re-declare to use helper
    std::unordered_set<Node*> visited_set;
    
    // Standard DFS Post-Order
    // We need a helper that takes shared_ptr
    struct DFS {
        static void run(std::shared_ptr<Node> node, std::unordered_set<Node*>& visited, std::vector<std::shared_ptr<Node>>& out) {
            if (visited.count(node.get())) return;
            visited.insert(node.get());
            
            for (const auto& edge : node->next_edges()) {
                if (edge.is_valid()) {
                    run(edge.function, visited, out);
                }
            }
            out.push_back(node);
        }
    };
    
    DFS::run(root_fn, visited_set, result);
    
    // result is now [Leaf, ..., Root]. We need [Root, ..., Leaf].
    std::reverse(result.begin(), result.end());
    
    return result;
}

void backward(const Tensor& root, const Tensor* grad_output) {
    // Validate
    if (!root.requires_grad()) {
        throw std::runtime_error("backward: tensor does not require gradients");
    }
    
    // For scalar tensors, grad_output can be omitted
    bool is_scalar = root.ndim() == 0 || root.numel() == 1;
    
    // Initialize root gradient
    Tensor root_grad;
    if (grad_output) {
        root_grad = *grad_output;
    } else if (is_scalar) {
        root_grad = Tensor::ones(root.shape(), TensorOptions()
            .with_dtype(root.dtype())
            .with_device(root.device()));
    } else {
        throw std::runtime_error(
            "backward: grad_output must be specified for non-scalar tensors");
    }
    
    // Topological sort
    auto nodes = topological_sort(root);
    
    // Gradient accumulation map
    std::unordered_map<Node*, std::vector<Tensor>> grad_map;
    
    // Initialize root gradient
    auto root_fn = root.grad_fn();
    if (root_fn) {
        grad_map[root_fn.get()] = {root_grad};
    } else {
        // Root is a leaf - accumulate directly
        if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(
                root.unsafeGetTensorImpl()->autograd_meta());
            if (meta->has_grad()) {
                // Accumulate
                Tensor& existing_grad = meta->mutable_grad(root.unsafeGetTensorImpl());
                Tensor new_grad = operator+(existing_grad, root_grad);
                meta->set_grad(new_grad);
            } else {
                meta->set_grad(root_grad);
            }
        }
        return;
    }
    
    // grad_map[root_fn.get()] = {root_grad};
    
    // OPTIMIZATION: Reserve typical capacity for grad vectors to avoid reallocations
    grad_map.reserve(nodes.size());

    // Process nodes in topological order
    for (const auto& node : nodes) {
        Node* node_ptr = node.get();
        
        // Check if this node has any gradients to process
        auto it = grad_map.find(node_ptr);
        if (it == grad_map.end() || it->second.empty()) {
            continue;  // No gradients for this node
        }
        
        // Sum all gradients for this node
        auto& node_grads = it->second;
        Tensor grad = node_grads[0];
        for (size_t i = 1; i < node_grads.size(); ++i) {
            grad = operator+(grad, node_grads[i]);
        }
        
        // Apply backward function (operator() handles hooks)
        std::vector<Tensor> input_grads = (*node_ptr)({grad});
        
        // MEMORY OPTIMIZATION: Release saved variables after backward to free memory
        node_ptr->release_saved_variables();
        
        // Also clear the processed grads to free them
        it->second.clear();
        
        // Distribute gradients to next edges
        const auto& edges = node_ptr->next_edges();
        for (size_t i = 0; i < edges.size() && i < input_grads.size(); ++i) {
            if (!edges[i].is_valid()) {
                continue;
            }
            
            auto next_fn = edges[i].function;
            auto& vec = grad_map[next_fn.get()];
            if (vec.empty()) vec.reserve(2);  // Pre-reserve for typical case
            vec.push_back(input_grads[i]);
        }
    }
        for (auto& [node_ptr, grads] : grad_map) {
        if (grads.empty()) continue;
        
        // Check if already processed (was in nodes list)
        bool was_processed = false;
        for (const auto& node : nodes) {
            if (node.get() == node_ptr) {
                was_processed = true;
                break;
            }
        }
        
        if (!was_processed) {
            // Sum gradients
            Tensor grad = grads[0];
            for (size_t i = 1; i < grads.size(); ++i) {
                grad = operator+(grad, grads[i]);
            }
            // Apply (this calls GradAccumulator::apply which sets grad_ in AutogradMeta)
            // operator() handles any node-level hooks registered on GradAccumulator
            (*node_ptr)({grad});
        }
    }
}

} // namespace autograd
} // namespace OwnTensor