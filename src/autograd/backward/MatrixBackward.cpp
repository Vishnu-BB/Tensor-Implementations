#include "autograd/backward/MatrixBackward.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/UnaryOps/Reduction.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor {
namespace autograd {

static Tensor reduce_to_shape(const Tensor& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) return grad;
    
    Tensor res = grad;
    int64_t grad_ndim = grad.ndim();
    int64_t target_ndim = target_shape.dims.size();
    
    std::vector<int64_t> dims_to_sum;
    
    // 1. Handle rank mismatch (leading dimensions)
    if (grad_ndim > target_ndim) {
        for (int64_t i = 0; i < grad_ndim - target_ndim; ++i) {
            dims_to_sum.push_back(i);
        }
    }
    
    // 2. Handle broadcasting in shared dimensions
    for (int64_t i = 0; i < target_ndim; ++i) {
        int64_t target_dim_idx = target_ndim - 1 - i;
        int64_t grad_dim_idx = grad_ndim - 1 - i;
        
        if (target_shape.dims[target_dim_idx] == 1 && grad.shape().dims[grad_dim_idx] > 1) {
            dims_to_sum.push_back(grad_dim_idx);
        }
    }
    
    if (!dims_to_sum.empty()) {
        res = reduce_sum(res, dims_to_sum, true);
    }
    
    if (res.shape() != target_shape) {
        res = res.reshape(target_shape);
    }
    
    return res;
}

MatmulBackward::MatmulBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> MatmulBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MatmulBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    Tensor b_t = saved_b_.t();
    
    Tensor grad_a = matmul(grad_output, b_t);
    grad_a = reduce_to_shape(grad_a, saved_a_.shape());
    
    Tensor grad_b;
    
    // Optimization for Linear Layer case: [Batch, T, Hidden] @ [Hidden, Out]
    // where we want to avoid materializing [Batch, Hidden, Out] before reduction
    if (saved_b_.ndim() == 2 && saved_a_.ndim() > 2) {
        int64_t hidden_dim = saved_a_.shape().dims.back();
        int64_t output_dim = grad_output.shape().dims.back();
        
        // Reshape [B, T, Hidden] -> [B*T, Hidden]
        Tensor a_flat = saved_a_.reshape(Shape{{-1, hidden_dim}});
        // Reshape [B, T, C] -> [B*T, C]
        Tensor g_flat = grad_output.reshape(Shape{{-1, output_dim}});
        
        // [Hidden, BT] @ [BT, C] -> [Hidden, C] (implicitly sums over B*T)
        grad_b = matmul(a_flat.t(), g_flat);
    } else {
        // General case
        Tensor a_t = saved_a_.t();
        grad_b = matmul(a_t, grad_output);
        grad_b = reduce_to_shape(grad_b, saved_b_.shape());
    }
    
    return {grad_a, grad_b};
}

} // namespace autograd
} // namespace OwnTensor