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
#include <iomanip>
#include <string>

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

void print_tensor_info(const std::string& label, const Tensor& t) {
    Tensor t_cpu = t.to_cpu();
    float mean_val = mean(t_cpu).data<float>()[0];
    std::cout << "  " << std::left << std::setw(25) << label 
              << " | Mean: " << std::fixed << std::setprecision(6) << mean_val;
            //   << " | Shape: " << t.shape() << "\n";
}

// --- Test Blocks ---

// 1. Divergent Paths: Input used inside and outside checkpoint
variable_list divergent_block(const variable_list& inputs) {
    return {mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f))};
}

void test_divergent_paths(Device device) {
    std::cout << "\n[TEST] Divergent Paths (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{2, 2}}, opts);
    
    // Path 1: Through checkpoint
    variable_list out_cp = checkpoint(divergent_block, {x});
    // Path 2: Outside checkpoint
    Tensor out_direct = mul(x, Tensor::full(x.shape(), opts, 3.0f));
    
    Tensor loss = add(sum(out_cp[0]), sum(out_direct));
    
    print_tensor_info("Forward CP Output", out_cp[0]);
    print_tensor_info("Forward Direct Output", out_direct);
    
    std::cout << "Running backward...\n";
    loss.backward();
    
    // Expected grad: 2.0 (from CP) + 3.0 (from Direct) = 5.0
    Tensor expected_grad = Tensor::full(Shape{{2, 2}}, opts, 5.0f);
    print_tensor_info("Input Grad", x.grad_view());
    
    assert(compare_tensors(x.grad_view(), expected_grad));
    std::cout << "✓ test_divergent_paths passed!\n";
}

// 2. Chained Checkpoints
variable_list block1(const variable_list& inputs) {
    return {add(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 1.0f))};
}
variable_list block2(const variable_list& inputs) {
    return {mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f))};
}

void test_chained_checkpoints(Device device) {
    std::cout << "\n[TEST] Chained Checkpoints (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{2, 2}}, opts);
    
    variable_list out1 = checkpoint(block1, {x});
    variable_list out2 = checkpoint(block2, {out1[0]});
    
    Tensor loss = sum(out2[0]);
    
    print_tensor_info("Forward Out1", out1[0]);
    print_tensor_info("Forward Out2", out2[0]);
    
    std::cout << "Running backward...\n";
    loss.backward();
    
    // (x + 1) * 2 = 2x + 2. Grad w.r.t x is 2.0
    Tensor expected_grad = Tensor::full(Shape{{2, 2}}, opts, 2.0f);
    print_tensor_info("Input Grad", x.grad_view());
    
    assert(compare_tensors(x.grad_view(), expected_grad));
    std::cout << "✓ test_chained_checkpoints passed!\n";
}

// 3. Shared Inputs
variable_list shared_input_block(const variable_list& inputs) {
    return {mul(inputs[0], inputs[1])};
}

void test_shared_inputs(Device device) {
    std::cout << "\n[TEST] Shared Inputs (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 3.0f);
    
    variable_list out = checkpoint(shared_input_block, {x, x});
    Tensor loss = sum(out[0]);
    
    print_tensor_info("Forward Output", out[0]);
    
    std::cout << "Running backward...\n";
    loss.backward();
    
    // x * x = x^2. Grad is 2x = 2 * 3 = 6.0
    Tensor expected_grad = Tensor::full(Shape{{2, 2}}, opts, 6.0f);
    print_tensor_info("Input Grad", x.grad_view());
    
    assert(compare_tensors(x.grad_view(), expected_grad));
    std::cout << "✓ test_shared_inputs passed!\n";
}

// 4. RNG Consistency (Stochastic)
variable_list stochastic_block(const variable_list& inputs) {
    Tensor noise = Tensor::rand<float>(inputs[0].shape(), inputs[0].opts());
    return {mul(inputs[0], noise)};
}

void test_rng_consistency(Device device) {
    std::cout << "\n[TEST] RNG Consistency (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{4, 4}}, opts);
    
    RNG::set_seed(42);
    variable_list out = checkpoint(stochastic_block, {x});
    Tensor loss = sum(out[0]);
    
    print_tensor_info("Forward Output", out[0]);
    
    std::cout << "Running backward...\n";
    loss.backward();
    
    Tensor grad_cp = x.grad_view().clone();
    print_tensor_info("Input Grad (CP)", grad_cp);
    
    // Baseline without checkpoint
    x.zero_grad();
    RNG::set_seed(42);
    variable_list out_base = stochastic_block({x});
    sum(out_base[0]).backward();
    Tensor grad_base = x.grad_view();
    print_tensor_info("Input Grad (Base)", grad_base);
    
    assert(compare_tensors(grad_cp, grad_base));
    std::cout << "✓ test_rng_consistency passed!\n";
}

// 5. Mixed Grad Requirements
variable_list mixed_grad_block(const variable_list& inputs) {
    return {add(inputs[0], inputs[1])};
}

void test_mixed_grad(Device device) {
    std::cout << "\n[TEST] Mixed Grad Requirements (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    Tensor x = Tensor::ones(Shape{{2, 2}}, TensorOptions().with_device(device).with_req_grad(true));
    Tensor y = Tensor::ones(Shape{{2, 2}}, TensorOptions().with_device(device).with_req_grad(false));
    
    variable_list out = checkpoint(mixed_grad_block, {x, y});
    Tensor loss = sum(out[0]);
    
    print_tensor_info("Forward Output", out[0]);
    
    std::cout << "Running backward...\n";
    loss.backward();
    
    assert(x.has_grad());
    assert(!y.has_grad());
    print_tensor_info("X Grad", x.grad_view());
    
    std::cout << "✓ test_mixed_grad passed!\n";
}

int main() {
    std::vector<Device> devices = {Device::CPU};
#ifdef WITH_CUDA
    devices.push_back(Device::CUDA);
#endif

    for (Device dev : devices) {
        try {
            test_divergent_paths(dev);
            test_chained_checkpoints(dev);
            test_shared_inputs(dev);
            test_rng_consistency(dev);
            test_mixed_grad(dev);
        } catch (const std::exception& e) {
            std::cerr << "Error on " << (dev == Device::CPU ? "CPU" : "GPU") << ": " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "\nAll production-ready checkpoint tests passed! ✓\n";
    return 0;
}