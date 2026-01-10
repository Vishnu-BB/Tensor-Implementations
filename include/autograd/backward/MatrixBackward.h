#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for matrix multiplication: a @ b
 * 
 * Forward: out = a @ b
 * Backward: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
 */
class MatmulBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
    
public:
    MatmulBackward(const Tensor& a, const Tensor& b);
    
    std::string name() const override { return "MatmulBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
