#include "Checkpointing/CheckpointNode.h"
#include "Checkpointing/GradMode.h"
#include "autograd/Engine.h"
#include "autograd/ops_template.h"
#include "autograd/operations/ReductionOps.h"
#include <stdexcept>
#include <iostream>

namespace OwnTensor {
namespace autograd {

CheckpointNode::CheckpointNode(
    std::function<variable_list(const variable_list&)> forward_fn,
    const variable_list& inputs,
    RNGState rng_state,
    size_t num_outputs)
    : Node(inputs.size()),
      forward_fn_(std::move(forward_fn)),
      rng_state_(std::move(rng_state)),
      num_outputs_(num_outputs) {
    
    saved_inputs_.reserve(inputs.size());
    input_requires_grad_.reserve(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        saved_inputs_.emplace_back(input, false);
        input_requires_grad_.push_back(input.requires_grad());
        
        if (input.requires_grad()) {
            Tensor& input_mut = const_cast<Tensor&>(input);
            set_next_edge(i, get_grad_edge(input_mut));
        } else {
            set_next_edge(i, Edge{});
        }
    }
}

variable_list CheckpointNode::apply(variable_list&& grads) {
    // std::cout << "[DEBUG] CheckpointNode::apply started. Grads size: " << grads.size() << "\n";
    
    // 1. Restore RNG state to ensure deterministic recomputation (e.g., Dropout).
    RNGStateGuard rng_guard;
    RNG::set_state(rng_state_);

    // 3. Unpack inputs and create views to isolate the recomputation graph.
    // This prevents the local backward pass from propagating into the global graph.
    variable_list recompute_inputs;
    recompute_inputs.reserve(saved_inputs_.size());
    for (size_t i = 0; i < saved_inputs_.size(); ++i) {
        const auto& sv = saved_inputs_[i];
        Tensor input = sv.unpack(shared_from_this());
        if (input.unsafeGetTensorImpl()) {
            // Create a view that shares storage but has its own AutogradMeta.
            Tensor input_view = input.view(input.shape()).detach();
            
            // Important: We set requires_grad on the view BEFORE enabling GradMode
            // so that it acts as a leaf in the local recomputation graph.
            input_view.set_requires_grad(input_requires_grad_[i]);
            recompute_inputs.push_back(input_view);
        } else {
            recompute_inputs.push_back(Tensor());
        }
    }

    // 4. Enable gradients for recomputation.
    // The initial forward pass was done in no_grad mode, so we must
    // re-enable it here to build the local computational graph.
    GradModeGuard grad_guard(true);
    // Before local backward:
    set_dependency_tracking_enabled(false); 
    // 5. Re-run forward pass to build the local graph.
    variable_list outputs = forward_fn_(recompute_inputs);

    if (outputs.size() != grads.size()) {
        throw std::runtime_error(
            "CheckpointNode::apply: Number of recomputed outputs (" + 
            std::to_string(outputs.size()) + ") does not match number of gradients (" + 
            std::to_string(grads.size()) + ")");
    }

    
    // 6. Disable dependency tracking for local graph
    set_dependency_tracking_enabled(false);
    
    // Set dependencies to a high value for local nodes to prevent eager release
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].requires_grad() && outputs[i].grad_fn()) {
            outputs[i].grad_fn()->set_dependencies(999999);
        }
    }

    // 7. Run local backward (with null checks)
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i].unsafeGetTensorImpl() || !grads[i].unsafeGetTensorImpl()) {
            continue;
        }
        if (outputs[i].requires_grad()) {
            outputs[i].backward(&grads[i]);
        }
    }

    // Re-enable dependency tracking
    set_dependency_tracking_enabled(true);

    // 8. Manually release after backward completes
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].requires_grad() && outputs[i].grad_fn()) {
            outputs[i].grad_fn()->release_saved_variables();
        }
    }

    // 9. Collect gradients for inputs
    variable_list input_grads;
    input_grads.reserve(recompute_inputs.size());
    for (size_t i = 0; i < recompute_inputs.size(); ++i) {
        if (input_requires_grad_[i] && recompute_inputs[i].has_grad()) {
            input_grads.push_back(recompute_inputs[i].grad_view().clone());
        } else {
            input_grads.push_back(Tensor());
        }
    }

    // 8. Clear saved data to free memory.
    // clear_saved_data();

    return input_grads;
}

// void CheckpointNode::clear_saved_data() {
//     // Release the forward function.
//     forward_fn_ = nullptr;
    
//     // Release the saved inputs.
//     for (auto& sv : saved_inputs_) {
//         sv.reset();
//     }
// }

} // namespace autograd
} // namespace OwnTensor