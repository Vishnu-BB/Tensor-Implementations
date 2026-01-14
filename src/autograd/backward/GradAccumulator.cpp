#include "autograd/backward/GradAccumulator.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace autograd {

GradAccumulator::GradAccumulator(TensorImpl* impl)
    : Node(1), leaf_impl_(impl) {}

std::vector<Tensor> GradAccumulator::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !leaf_impl_) {
        return {};
    }
    
    const Tensor& grad_output = grads[0];
    
    // Accumulate gradient into leaf tensor
    if (leaf_impl_->has_autograd_meta()) {
        auto* meta = static_cast<AutogradMeta*>(leaf_impl_->autograd_meta());
        
        if (meta->has_grad()) {
            // Accumulate: existing_grad += grad_output
            Tensor& existing_grad = meta->mutable_grad(leaf_impl_);
            Tensor new_grad = operator+(existing_grad, grad_output);
            meta->set_grad(new_grad);
        } else {
            // First gradient: just set it
            meta->set_grad(grad_output);
        }
        
        // Trigger post-accumulation hooks specifically after accumulation is done
        // for this backward pass. Since Engine.cpp sums all gradients for GradAccumulator
        // before calling apply, this is the final gradient for this pass.
        meta->trigger_post_acc_hooks(meta->grad());
    }
    
    // No outputs (leaf node)
    return {};
}

} // namespace autograd
} // namespace OwnTensor
