#include "autograd/backward/EmbeddingBackward.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <stdexcept>
#include <cstring>

namespace OwnTensor {
namespace autograd {

EmbeddingBackward::EmbeddingBackward(const Tensor& indices, int64_t num_embeddings, int padding_idx)
    : Node(1), saved_indices_(indices), num_embeddings_(num_embeddings), padding_idx_(padding_idx) {}

std::vector<Tensor> EmbeddingBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("EmbeddingBackward: no gradients provided");
    }

    const Tensor& grad_output = grads[0];
    // grad_output shape: [N, C] where N is number of indices, C is embedding dim
    // weight shape: [num_embeddings, C]
    
    int64_t N = saved_indices_.numel();
    int64_t C = grad_output.shape().dims[grad_output.ndim() - 1];
    
    // Create grad_weight with same options as grad_output but different shape
    Tensor grad_weight = Tensor::zeros(Shape{{num_embeddings_, C}}, 
        TensorOptions().with_device(grad_output.device()).with_dtype(grad_output.dtype()));
    
    // CPU implementation
    if (grad_output.is_cpu()) {
        float* g_w_ptr = grad_weight.data<float>();
        const float* g_o_ptr = grad_output.data<float>();
        const uint16_t* idx_ptr = saved_indices_.data<uint16_t>();
        
        for (int64_t n = 0; n < N; ++n) {
            uint16_t tok = idx_ptr[n];
            if (tok == padding_idx_) continue;
            
            if (tok >= num_embeddings_) {
                throw std::runtime_error("EmbeddingBackward: token id out of range");
            }
            
            float* row = g_w_ptr + static_cast<size_t>(tok) * C;
            const float* go = g_o_ptr + n * C;
            
            for (int64_t j = 0; j < C; ++j) {
                row[j] += go[j];
            }
        }
    } else if (grad_output.is_cuda()) {
        cuda::embedding_backward_cuda(
            saved_indices_.data<uint16_t>(),
            grad_output.data<float>(),
            grad_weight.data<float>(),
            N, C, num_embeddings_, padding_idx_
        );
    } else {
        throw std::runtime_error("EmbeddingBackward: unsupported device");
    }
    
    return {grad_weight};
}

} // namespace autograd
} // namespace OwnTensor