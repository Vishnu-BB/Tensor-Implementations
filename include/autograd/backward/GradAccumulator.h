#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"

namespace OwnTensor {
namespace autograd {

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
