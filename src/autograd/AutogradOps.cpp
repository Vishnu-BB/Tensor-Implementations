#include "autograd/AutogradOps.h"
#include "autograd/Functions.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ConditionalOps.h"

namespace OwnTensor {
namespace autograd {

Tensor add(const Tensor& a, const Tensor& b) {
    // Forward pass
    Tensor result = operator+(a, b);
    
    // Build graph if needed
    if (a.requires_grad() || b.requires_grad()) {
        auto grad_fn = std::make_shared<AddBackward>();
        
        // Set up edges to inputs
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, make_edge(a.grad_fn(), a.output_nr()));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, make_edge(b.grad_fn(), b.output_nr()));
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
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, make_edge(a.grad_fn(), a.output_nr()));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, make_edge(b.grad_fn(), b.output_nr()));
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
        if (a.requires_grad()) {
            grad_fn->set_next_edge(0, make_edge(a.grad_fn(), a.output_nr()));
        }
        if (b.requires_grad()) {
            grad_fn->set_next_edge(1, make_edge(b.grad_fn(), b.output_nr()));
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
        grad_fn->set_next_edge(0, make_edge(x.grad_fn(), x.output_nr()));
        
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
        grad_fn->set_next_edge(0, make_edge(x.grad_fn(), x.output_nr()));
        
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
        grad_fn->set_next_edge(0, make_edge(x.grad_fn(), x.output_nr()));
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

} // namespace autograd
} // namespace OwnTensor
