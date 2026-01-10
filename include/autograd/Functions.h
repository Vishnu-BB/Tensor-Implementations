#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include <memory>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for addition: a + b
 * 
 * Forward: out = a + b
 * Backward: grad_a = grad_out, grad_b = grad_out
 */
class AddBackward : public Node {
public:
    AddBackward() : Node(2) {}  // 2 inputs
    
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

/**
 * @brief Backward function for ReLU: max(0, x)
 * 
 * Forward: out = max(0, x)
 * Backward: grad_x = grad_out * (x > 0)
 */
class ReluBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    explicit ReluBackward(const Tensor& input);
    
    std::string name() const override { return "ReluBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for sum reduction
 * 
 * Forward: out = sum(x)
 * Backward: grad_x = grad_out (broadcasted to x.shape)
 */
class SumBackward : public Node {
private:
    Shape input_shape_;
    
public:
    explicit SumBackward(const Shape& input_shape);
    
    std::string name() const override { return "SumBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for mean reduction
 * 
 * Forward: out = mean(x)
 * Backward: grad_x = grad_out / numel(x)
 */
class MeanBackward : public Node {
private:
    Shape input_shape_;
    int64_t numel_;
    
public:
    MeanBackward(const Shape& input_shape, int64_t numel);
    
    std::string name() const override { return "MeanBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Gradient accumulator for leaf tensors
 * 
 * This node accumulates gradients into a leaf tensor's AutogradMeta.
 * It's used as the terminal node in the backward graph for parameters.
 */
class GradAccumulator : public Node {
private:
    TensorImpl* leaf_impl_;  // Non-owning pointer to leaf tensor's impl
    
public:
    explicit GradAccumulator(TensorImpl* impl);
    
    std::string name() const override { return "GradAccumulator"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
