#include "autograd/operations/ActivationOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ActivationBackward.h"
#include "ops/TensorOps.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Trigonometry.h"
#include <cmath>

namespace OwnTensor {
namespace autograd {

Tensor relu(const Tensor& x) {
    return make_unary_op<ReluBackward>(x,
        [](const Tensor& input) {
            Tensor zero = Tensor::zeros(input.shape(), 
                TensorOptions().with_dtype(input.dtype()).with_device(input.device()));
            return where(input > zero, input, zero);
        },
        x);  // Pass x to ReluBackward constructor
}

Tensor gelu(const Tensor& x) {
    return make_unary_op<GeLUBackward>(x,
        [](const Tensor& input) {
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            Tensor half_x = 0.5f * input;
            Tensor x_cubed = input * input * input;
            Tensor tanh_inp = sqrt_2_over_pi * (input - 0.044715f * x_cubed);
            Tensor inner_output = 1.0f + tanh(tanh_inp);
            return half_x * inner_output;
        },
        x);  // Pass x to GeLUBackward constructor
}

Tensor sigmoid(const Tensor& x) {
    // Compute forward and save output for backward
    Tensor exp_input = exp(x);
    Tensor denom = 1.0f + exp_input;
    Tensor output = exp_input / denom;
    
    // Build graph if needed
    if (x.requires_grad()) {
        auto grad_fn = std::make_shared<SigmoidBackward>(output);
        Tensor& x_mut = const_cast<Tensor&>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    
    return output;
}

Tensor softmax(const Tensor& x, int64_t dim) {
    int64_t ndim = x.ndim();
    if (dim < 0) dim += ndim;
    
    // Forward: exp(x - max(x)) / sum(exp(x - max(x)))
    Tensor max_val = reduce_max(x, {dim}, true);
    Tensor shifted = x - max_val;
    Tensor exp_x = exp(shifted);
    Tensor sum_exp = reduce_sum(exp_x, {dim}, true);
    Tensor output = exp_x / sum_exp;
    
    // Build graph if needed
    if (x.requires_grad()) {
        auto grad_fn = std::make_shared<SoftmaxBackward>(output, dim);
        Tensor& x_mut = const_cast<Tensor&>(x);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    
    return output;
}

} // namespace autograd
} // namespace OwnTensor
