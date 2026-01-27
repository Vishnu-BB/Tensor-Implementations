#include "autograd/backward/GradAccumulator.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace autograd {

GradAccumulator::GradAccumulator(TensorImpl* impl)
    : Node(1), leaf_impl_(impl) {}

std::vector<GradAccumulator*> GradAccumulator::pool_;
std::mutex GradAccumulator::pool_mutex_;

void GradAccumulator::reset(TensorImpl* impl) {
    leaf_impl_ = impl;
    // Reset Node state if necessary (clearing edges, hooks etc.)
    // For now assuming Node state is clean or doesn't matter for new usage as leaf
    // Important: Node construction increments sequence_nr. Reuse means sequence_nr is stale?
    // Engine uses topological sort which re-computes dependencies. 
    // Sequence nr is mostly for debug or deterministic ties.
    // Ideally we should re-assign a new sequence number.
    // Accessing protected member in Node? 
    // We can just leave it. If Engine relies heavily on strict increasing seq number for correctness it might issue.
    // Engine uses topological sort based on structure, sequence_nr is secondary.
    clear_edges(); 
    // Reset edge to empty/invalid if any
}

std::shared_ptr<GradAccumulator> GradAccumulator::make(TensorImpl* impl) {
    GradAccumulator* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        if (!pool_.empty()) {
            ptr = pool_.back();
            pool_.pop_back();
        }
    }
    
    if (!ptr) {
        ptr = new GradAccumulator(impl);
    } else {
        ptr->reset(impl);
    }
    
    // Return shared_ptr with custom deleter that returns to pool
    return std::shared_ptr<GradAccumulator>(ptr, [](GradAccumulator* p) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pool_.push_back(p);
    });
}

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
