#include "autograd/operations/MatrixOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/MatrixBackward.h"
#include "autograd/backward/EmbeddingBackward.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <algorithm>

namespace OwnTensor {
namespace autograd {

Tensor matmul(const Tensor& a, const Tensor& b) {
    return make_binary_op<MatmulBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return OwnTensor::matmul(x, y); },
        a, b);  // Pass a, b to MatmulBackward constructor
}

Tensor embedding(const Tensor& indices, const Tensor& weight, int padding_idx) {
    auto forward_op = [indices, padding_idx](const Tensor& w) {
        int64_t N = indices.numel();
        int64_t C = w.shape().dims[1]; // Embedding dimension
        int64_t V = w.shape().dims[0]; // Vocab size
        
        // Output shape matches indices shape + embedding dimension
        std::vector<int64_t> out_dims = indices.shape().dims;
        out_dims.push_back(C);
        Shape out_shape(out_dims);

        Tensor result = Tensor::zeros(out_shape, 
            TensorOptions().with_device(w.device()).with_dtype(w.dtype()));
        
        if (w.is_cpu()) {
            const uint16_t* idx_ptr = indices.data<uint16_t>();
            const float* w_ptr = w.data<float>();
            float* res_ptr = result.data<float>();
            
            for (int64_t i = 0; i < N; ++i) {
                uint16_t tok = idx_ptr[i];
                if (tok == (uint16_t)padding_idx) continue;
                
                if (tok >= V) {
                    throw std::runtime_error("embedding: token id out of range");
                }
                
                const float* row = w_ptr + (size_t)tok * C;
                float* res_row = res_ptr + (size_t)i * C;
                std::copy(row, row + C, res_row);
            }
        } else if (w.is_cuda()) {
            cuda::embedding_forward_cuda(
                indices.data<uint16_t>(),
                w.data<float>(),
                result.data<float>(),
                N, C, V, padding_idx
            );
        } else {
             throw std::runtime_error("embedding: unsupported device");
        }
        
        return result;
    };
    
    // We treat this as a unary op on 'weight' because 'indices' doesn't require grad
    return make_unary_op<EmbeddingBackward>(weight, forward_op, indices, weight.shape().dims[0], padding_idx);
}

} // namespace autograd
} // namespace OwnTensor