#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for MSE loss: mean((pred - target)^2)
 * 
 * Backward: grad_pred = 2 * (pred - target) / numel
 */
class MSELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    MSELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    std::string name() const override { return "MSELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for MAE loss: mean(|pred - target|)
 * 
 * Backward: grad_pred = sign(pred - target) / numel
 */
class MAELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    MAELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    std::string name() const override { return "MAELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for binary cross entropy loss
 * 
 * Backward: grad_pred = -target/pred + (1-target)/(1-pred)
 */
class BCELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    BCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    std::string name() const override { return "BCELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for categorical cross entropy loss
 * 
 * Backward: grad_pred = -target / pred
 */
class CCELossBackward : public Node {
private:
    Tensor saved_pred_;
    Tensor saved_target_;
    int64_t numel_;
    
public:
    CCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel);
    
    std::string name() const override { return "CCELossBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for sparse cross entropy with logits
 * 
 * Backward: grad_logits = softmax(logits) - one_hot(target)
 */
class SparseCrossEntropyBackward : public Node {
private:
    Tensor saved_logits_;
    Tensor saved_target_;
    int64_t dim_;
    
public:
    SparseCrossEntropyBackward(const Tensor& logits, const Tensor& target, int64_t dim = -1);
    
    std::string name() const override { return "SparseCrossEntropyBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
