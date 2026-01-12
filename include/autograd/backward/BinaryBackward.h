#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for addition: a + b
 * 
 * Forward: out = a + b
 * Backward: grad_a = grad_out, grad_b = grad_out
 */
class AddBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
public:
    AddBackward(const Tensor& a, const Tensor& b);
    
    std::string name() const override { return "AddBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for multiplication: a * b
 * 
 * Forward: out = a * b
 * Backward: grad_a = grad_out * b, grad_b = grad_out * a
 */
class MulBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
    
public:
    MulBackward(const Tensor& a, const Tensor& b);
    
    std::string name() const override { return "MulBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for subtraction: a - b
 * 
 * Forward: out = a - b
 * Backward: grad_a = grad_out, grad_b = -grad_out
 */
class SubBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
public:
    SubBackward(const Tensor& a, const Tensor& b);
    
    std::string name() const override { return "SubBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for division: a / b
 * 
 * Forward: out = a / b
 * Backward: grad_a = grad_out / b, grad_b = -grad_out * a / b^2
 */
class DivBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
    
public:
    DivBackward(const Tensor& a, const Tensor& b);
    
    std::string name() const override { return "DivBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
