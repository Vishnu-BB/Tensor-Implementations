
#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "core/RNG.h"
#include "nn/NN.h"
#include <iostream>
#include <unistd.h>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

void marker(const std::string& label) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "Active Tensors: " << Tensor::get_active_tensor_count() << std::endl;
    sleep(1);
}

// Deep computation (20 layers)
Tensor deep_computation(const Tensor& input) {
    Tensor x = input;
    TensorOptions opts = x.opts();
    
    for (int i = 0; i < 20; ++i) {
        x = tanh(x);
        Tensor ones = Tensor::ones(x.shape(), opts);
        ones.fill(0.01f);
        x = add(x, ones);
    }
    
    return x;
}

int main() {
    std::cout << "=== WITHOUT Checkpointing (Baseline) ===" << std::endl;
    
    Device device = Device::CPU;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    marker("START");
    
    // Create input
    Tensor input = Tensor::randn<float>(Shape{{256, 1024}}, opts);
    marker("After Input Creation");
    
    // Forward pass - stores ALL intermediates
    std::cout << "Running forward pass (storing intermediates)..." << std::endl;
    Tensor output = deep_computation(input);
    marker("After Forward - Intermediates STORED in memory");
    
    // Compute loss
    Tensor loss = mean(mul(output, output));
    std::cout << "Loss: " << *loss.data<float>() << std::endl;
    marker("After Loss Computation");
    
    // Backward pass
    std::cout << "Running backward pass..." << std::endl;
    loss.backward();
    marker("After Backward - PEAK MEMORY");
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Check Massif for peak memory usage" << std::endl;
    std::cout << "Before cleanup: " << Tensor::get_active_tensor_count() << std::endl;
    
    // Explicitly release all tensors
    input = Tensor();
    output = Tensor();
    loss = Tensor();
    std::cout << "After cleanup: " << Tensor::get_active_tensor_count() << std::endl;
    return 0;
}