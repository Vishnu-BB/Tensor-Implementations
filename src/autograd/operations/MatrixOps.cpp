#include "autograd/operations/MatrixOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/MatrixBackward.h"
#include "autograd/backward/LinearBackward.h"
#ifdef WITH_CUDA
#include "ops/LinearKernels.cuh"
#endif
#include "ops/Kernels.h"
#include "ops/TensorOps.h"
#include <algorithm>
#include "Checkpointing/GradMode.h"

namespace OwnTensor {
namespace autograd {

Tensor matmul(const Tensor& a, const Tensor& b) {
    return make_binary_op<MatmulBackward>(a, b,
        [](const Tensor& x, const Tensor& y) {
            NoGradGuard guard;
            return OwnTensor::matmul(x, y);
        },
        a, b);  // Pass a, b to MatmulBackward constructor
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // Forward: input @ weight + bias (bias handled by Tensor ops, not autograd ops)
    auto forward_fn = [](const Tensor& x, const Tensor& w, const Tensor& b) {
#ifdef WITH_CUDA
        if (x.is_cuda()) {
             Tensor out = Tensor::empty(Shape{{x.shape().dims[0], w.shape().dims[1]}}, x.opts()); 
             // Shape inference: x [..., In], w [In, Out]?
             // Actually, assuming w is correct for matmul(x,w).
             // We need to be careful about shape inference.
             // Best to utilize `cuda_linear_forward`'s logic or internal matmul helpers but we don't return new tensor there easily.
             
             // Since `cuda_linear_forward` in my implementation (defined below/above) 
             // does `output = matmul(x, w); add_bias(output, b);` internally,
             // we can just call it passing a dummy output tensor reference to be assigned, 
             // OR modify `cuda_linear_forward` to RETURN a tensor.
             
             // The implementation I wrote accepts `Tensor& output`.
             Tensor output; 
             cuda_linear_forward(x, w, b, output);
             return output;
        }
#endif
        // We use raw Tensor operations here, not autograd wrappers
        // to avoid creating intermediate nodes
        Tensor out = OwnTensor::matmul(x, w);
        if (b.is_valid()) {
            out = out + b;
        }
        return out;
    };
    
    // Create output tensor with attached grad_fn
    Tensor result = forward_fn(input, weight, bias);
    
    if (input.requires_grad() || weight.requires_grad()) {
        auto grad_fn = std::make_shared<LinearBackward>(input, weight);
        
        // Connect edges
        Tensor& input_mut = const_cast<Tensor&>(input);
        Tensor& weight_mut = const_cast<Tensor&>(weight);
        Tensor& bias_mut = const_cast<Tensor&>(bias);

        if (input.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(input_mut));
        }
        if (weight.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(weight_mut));
        }
        if (bias.is_valid() && bias.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(bias_mut));
        }
        
        result.set_grad_fn(grad_fn);
    }
    
    return result;
}



} // namespace autograd
} // namespace OwnTensor