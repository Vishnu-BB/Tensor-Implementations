#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for embedding lookup.
 * 
 * Forward: out[i] = weight[indices[i]]
 * Backward: grad_weight[indices[i]] += grad_output[i]
 */
class EmbeddingBackward : public Node {
private:
    Tensor saved_indices_;
    int64_t num_embeddings_;
    int padding_idx_;
    
public:
    EmbeddingBackward(const Tensor& indices, int64_t num_embeddings, int padding_idx = -1);
    
    std::string name() const override { return "EmbeddingBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor