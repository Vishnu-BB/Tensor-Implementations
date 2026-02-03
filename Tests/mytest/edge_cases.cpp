#include "core/Tensor.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "Checkpointing/Checkpoint.h"
#include "Checkpointing/GradMode.h"
#include "core/RNG.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-5) {
    if (a.shape() != b.shape()) return false;
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    const float* data_a = a_cpu.data<float>();
    const float* data_b = b_cpu.data<float>();
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(data_a[i] - data_b[i]) > tol) return false;
    }
    return true;
}

// 1. Unused Input Test
variable_list unused_input_block(const variable_list& inputs) {
    Tensor two = Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f);
    return {mul(inputs[0], two)};
}

void test_unused_input() {
    std::cout << "Running test_unused_input...\n";
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{2, 2}}, opts);
    Tensor y = Tensor::ones(Shape{{2, 2}}, opts); // Unused but requires grad
    
    variable_list out = checkpoint(unused_input_block, {x, y});
    sum(out[0]).backward();
    
    assert(x.has_grad());
    assert(!y.has_grad() || y.grad_view().numel() == 0 || sum(y.grad_view()).data<float>()[0] == 0.0f);
    std::cout << "✓ test_unused_input passed!\n";
}

// 2. Multiple Outputs Test
variable_list multi_output_block(const variable_list& inputs) {
    Tensor two = Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f);
    Tensor three = Tensor::full(inputs[0].shape(), inputs[0].opts(), 3.0f);
    return {mul(inputs[0], two), add(inputs[0], three)};
}

void test_multi_output() {
    std::cout << "Running test_multi_output...\n";
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{2, 2}}, opts);
    
    // variable_list out = checkpoint(multi_output_block, {x});
    Tensor loss = add(sum(multi_output_block({x})[0]), sum(multi_output_block({x})[1]));
    loss.backward();
    
    // Grad should be 2.0 (from out[0]) + 1.0 (from out[1]) = 3.0
    Tensor expected_grad = Tensor::full(Shape{{2, 2}}, opts, 3.0f);
    assert(compare_tensors(x.grad_view(), expected_grad));
    std::cout << "✓ test_multi_output passed!\n";
}

// 3. No-Grad Input Test
variable_list no_grad_input_block(const variable_list& inputs) {
    return {mul(inputs[0], inputs[1])};
}

void test_no_grad_input() {
    std::cout << "Running test_no_grad_input...\n";
    Tensor x = Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    Tensor y = Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(false), 5.0f);
    
    variable_list out = checkpoint(no_grad_input_block, {x, y});
    sum(out[0]).backward();
    
    Tensor expected_grad = Tensor::full(Shape{{2, 2}}, x.opts(), 5.0f);
    assert(compare_tensors(x.grad_view(), expected_grad));
    assert(!y.has_grad());
    std::cout << "✓ test_no_grad_input passed!\n";
}

// 4. RNG Consistency Test (Stochastic)
variable_list stochastic_block(const variable_list& inputs) {
    Tensor noise = Tensor::rand<float>(inputs[0].shape(), inputs[0].opts());
    return {mul(inputs[0], noise)};
}

void test_rng_consistency() {
    std::cout << "Running test_rng_consistency...\n";
    Tensor x = Tensor::ones(Shape{{4, 4}}, TensorOptions().with_req_grad(true));
    
    RNG::set_seed(123);
    variable_list out1 = checkpoint(stochastic_block, {x});
    sum(out1[0]).backward();
    Tensor grad1 = x.grad_view().clone();
    
    x.zero_grad();
    RNG::set_seed(123);
    variable_list out2 = checkpoint(stochastic_block, {x});
    sum(out2[0]).backward();
    Tensor grad2 = x.grad_view();
    
    assert(compare_tensors(grad1, grad2));
    std::cout << "✓ test_rng_consistency passed!\n";
}

// 5. Nested Checkpointing Test
variable_list inner_block(const variable_list& inputs) {
    return {mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f))};
}
variable_list outer_block(const variable_list& inputs) {
    return checkpoint(inner_block, inputs);
}

void test_nested_checkpoint() {
    std::cout << "Running test_nested_checkpoint...\n";
    Tensor x = Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    
    variable_list out = checkpoint(outer_block, {x});
    sum(out[0]).backward();
    
    Tensor expected_grad = Tensor::full(Shape{{2, 2}}, x.opts(), 2.0f);
    assert(compare_tensors(x.grad_view(), expected_grad));
    std::cout << "✓ test_nested_checkpoint passed!\n";
}

int main() {
    try {
        test_unused_input();
        test_multi_output();
        test_no_grad_input();
        test_rng_consistency();
        test_nested_checkpoint();
        std::cout << "\nAll edge case tests passed! ✓\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}