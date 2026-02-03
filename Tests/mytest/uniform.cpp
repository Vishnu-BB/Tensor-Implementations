#include "core/Tensor.h"
#include "nn/NN.h"
#include "Checkpointing/Checkpoint.h"
#include "Checkpointing/GradMode.h"
#include "autograd/operations/ReductionOps.h"
#include "core/RNG.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace OwnTensor;
using namespace OwnTensor::autograd;
using namespace OwnTensor::nn;

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

void test_uniform_checkpointing(Device device) {
    std::cout << "\n[TEST] Uniform Checkpointing (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    // 1. Create a deep sequential model
    auto model = std::make_shared<Sequential>(std::initializer_list<Module*>{
        new Linear(10, 20), new ReLU(),
        new Linear(20, 20), new ReLU(),
        new Linear(20, 20), new ReLU(),
        new Linear(20, 20), new ReLU(),
        new Linear(20, 10)
    });
    model->to(device == Device::CPU ? DeviceIndex(Device::CPU) : DeviceIndex(Device::CUDA));

    Tensor x = Tensor::randn<float>(Shape{{4, 10}}, opts);
    
    // --- Baseline: No Checkpointing ---
    RNG::set_seed(42);
    // Reset parameters to ensure same initialization if we were to re-init, 
    // but here we just use the same model instance.
    // We need to clone the model or its parameters to compare.
    
    // Actually, let's just run forward/backward once, save grads, then zero grads and run with CP.
    
    std::cout << "Running baseline (no checkpointing)...\n";
    Tensor out_base = model->forward(x);
    Tensor loss_base = sum(out_base);
    loss_base.backward();
    
    std::vector<Tensor> grads_base;
    for (const auto& p : model->parameters()) {
        grads_base.push_back(p.grad_view().clone());
    }
    Tensor x_grad_base = x.grad_view().clone();
    
    // --- Test: With Checkpointing ---
    model->zero_grad();
    x.zero_grad();
    
    std::cout << "Running with checkpoint_sequential (2 segments)...\n";
    variable_list out_cp = checkpoint_sequential(model, 2, {x});
    Tensor loss_cp = sum(out_cp[0]);
    loss_cp.backward();
    
    // --- Verification ---
    std::cout << "Verifying gradients...\n";
    assert(compare_tensors(out_base, out_cp[0]));
    assert(compare_tensors(x_grad_base, x.grad_view()));
    
    auto params = model->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        assert(compare_tensors(grads_base[i], params[i].grad_view()));
    }
    
    std::cout << "✓ test_uniform_checkpointing (2 segments) passed!\n";

    // --- Test: With Checkpointing (5 segments) ---
    model->zero_grad();
    x.zero_grad();
    
    std::cout << "Running with checkpoint_sequential (5 segments)...\n";
    out_cp = checkpoint_sequential(model, 5, {x});
    sum(out_cp[0]).backward();
    
    assert(compare_tensors(x_grad_base, x.grad_view()));
    for (size_t i = 0; i < params.size(); ++i) {
        assert(compare_tensors(grads_base[i], params[i].grad_view()));
    }
    std::cout << "✓ test_uniform_checkpointing (5 segments) passed!\n";
}

int main() {
    try {
        test_uniform_checkpointing(Device::CPU);
#ifdef WITH_CUDA
        test_uniform_checkpointing(Device::CUDA);
#endif
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll uniform checkpointing tests passed! ✓\n";
    return 0;
}