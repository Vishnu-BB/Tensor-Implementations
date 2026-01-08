#include "autograd/AutogradOps.h"
#include "autograd/Functions.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ConditionalOps.h"
#include "core/AutogradMeta.h"

namespace OwnTensor {
namespace autograd {

// Helper: Get edge to input tensor, creating/reusing GradAccumulator for leaves
static Edge get_grad_edge(Tensor& tensor) {
    if (tensor.grad_fn()) {
        // Non-leaf: connect to existing grad_fn
        return make_edge(tensor.grad_fn(), tensor.output_nr());
    } else if (tensor.requires_grad()) {
        // Leaf: get or create GradAccumulator from AutogradMeta
        TensorImpl* impl = tensor.unsafeGetTensorImpl();
        if (impl->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(impl->autograd_meta());
            
            // Try to get existing accumulator
            auto accumulator = meta->grad_accumulator_.lock();
            if (!accumulator) {
                // Create new one and cache it
                accumulator = std::make_shared<GradAccumulator>(impl);
                meta->grad_accumulator_ = accumulator;
            }
            return make_edge(accumulator, 0);
        }
    }
    return Edge{};  // No gradient needed
}

Tensor add(const Tensor& a, const Tensor& b) {
    // Forward pass
    Tensor result = operator+(a, b);
    
    // Build graph if needed
    if (a.requires_grad() || b.requires_grad()) {
        auto grad_fn = std::make_shared<AddBackward>();
        
        // Set up edges to inputs (mutable refs needed for GradAccumulator)
        Tensor& a_mut = const_cast<Tensor&>(a);
        Tensor& b_mut = const_cast<Tensor&>(b);
        
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(a_mut));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(b_mut));
        }
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // Forward pass
    Tensor result = operator*(a, b);
    
    // Build graph if needed
    if (a.requires_grad() || b.requires_grad()) {
        auto grad_fn = std::make_shared<MulBackward>(a, b);
        
        // Set up edges to inputs
        Tensor& a_mut = const_cast<Tensor&>(a);
        Tensor& b_mut = const_cast<Tensor&>(b);
        
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(a_mut));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(b_mut));
        }
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Forward pass
    Tensor result = OwnTensor::matmul(a, b);
    
    // Build graph if needed
    if (a.requires_grad() || b.requires_grad()) {
        auto grad_fn = std::make_shared<MatmulBackward>(a, b);
        
        // Set up edges to inputs
        Tensor& a_mut = const_cast<Tensor&>(a);
        Tensor& b_mut = const_cast<Tensor&>(b);
        
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(a_mut));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(b_mut));
        }
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor relu(const Tensor& x) {
    // Forward pass: max(0, x)
    Tensor zero = Tensor::zeros(x.shape(), TensorOptions().with_dtype(x.dtype()).with_device(x.device()));
    Tensor result = where(x > zero, x, zero);
    
    // Build graph if needed
    if (x.requires_grad()) {
        auto grad_fn = std::make_shared<ReluBackward>(x);
        
        // Set up edge to input
        Tensor& x_mut = const_cast<Tensor&>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor sum(const Tensor& x) {
    // Forward pass
    Tensor result = reduce_sum(x);
    
    // Build graph if needed
    if (x.requires_grad()) {
        auto grad_fn = std::make_shared<SumBackward>(x.shape());
        
        // Set up edge to input
        Tensor& x_mut = const_cast<Tensor&>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor mean(const Tensor& x) {
    // Forward pass
    Tensor result = reduce_mean(x);
    
    // Build graph if needed
    if (x.requires_grad()) {
        auto grad_fn = std::make_shared<MeanBackward>(x.shape(), x.numel());
        
        // Set up edge to input
        Tensor& x_mut = const_cast<Tensor&>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

} // namespace autograd
} // namespace OwnTensor
