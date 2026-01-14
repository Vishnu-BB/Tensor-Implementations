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

std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root) {
    std::vector<std::shared_ptr<Node>> result;
    std::unordered_set<Node*> visited;
    std::unordered_map<Node*, int> in_degree;
    
    // Get root's grad_fn
    auto root_fn = root.grad_fn();
    if (!root_fn) {
        return result;  // Leaf tensor, no graph to traverse
    }
    
    // BFS to build in-degree map
    std::queue<std::shared_ptr<Node>> queue;
    queue.push(root_fn);
    visited.insert(root_fn.get());
    in_degree[root_fn.get()] = 0;
    
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();
        
        for (const auto& edge : node->next_edges()) {
            if (edge.is_valid()) {
                auto next_node = edge.function;
                if (visited.find(next_node.get()) == visited.end()) {
                    visited.insert(next_node.get());
                    queue.push(next_node);
                    in_degree[next_node.get()] = 0;
                }
                in_degree[next_node.get()]++;
            }
        }
    }
    
    // Kahn's algorithm for topological sort
    std::queue<std::shared_ptr<Node>> zero_in_degree;
    zero_in_degree.push(root_fn);
    
    while (!zero_in_degree.empty()) {
        auto node = zero_in_degree.front();
        zero_in_degree.pop();
        result.push_back(node);
        
        for (const auto& edge : node->next_edges()) {
            if (edge.is_valid()) {
                auto next_node = edge.function;
                in_degree[next_node.get()]--;
                if (in_degree[next_node.get()] == 0) {
                    zero_in_degree.push(next_node);
                }
            }
        }
    }
    
    // Reverse for backward order (leaves to root)
    // std::reverse(result.begin(), result.end());  // already done by kahns alg itself 
    
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
    
    grad_map[root_fn.get()] = {root_grad};

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
        
        // Distribute gradients to next edges
        const auto& edges = node_ptr->next_edges();
        for (size_t i = 0; i < edges.size() && i < input_grads.size(); ++i) {
            if (!edges[i].is_valid()) {
                continue;
            }
            
            auto next_fn = edges[i].function;
            grad_map[next_fn.get()].push_back(input_grads[i]);
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