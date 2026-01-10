#include "autograd/backward/MatrixBackward.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

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
    Tensor a_t = saved_a_.t();
    
    Tensor grad_a = matmul(grad_output, b_t);
    Tensor grad_b = matmul(a_t, grad_output);
    
    return {grad_a, grad_b};
}

} // namespace autograd
} // namespace OwnTensor
