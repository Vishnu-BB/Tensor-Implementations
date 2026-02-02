/**
 * @file checkpoint_memory_WITH.cpp
 * @brief Test: Forward/backward WITH checkpointing
 * 
 * Run this second to see memory savings:
 *   valgrind --tool=massif --massif-out-file=with_checkpoint.out ./snippet_runner
 *   
 * Then compare the two Massif outputs:
 *   massif-visualizer without_checkpoint.out with_checkpoint.out
 */

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "Checkpointing/Checkpoint.h"
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

// Deep computation (20 layers) - same as WITHOUT version
variable_list deep_computation_checkpointed(const variable_list& inputs) {
    Tensor x = inputs[0];
    TensorOptions opts = x.opts();
    
    for (int i = 0; i < 20; ++i) {
        x = tanh(x);
        Tensor ones = Tensor::ones(x.shape(), opts);
        ones.fill(0.01f);
        x = add(x, ones);
    }
    
    return {x};
}

int main() {
    std::cout << "=== WITH Checkpointing (Memory Optimized) ===" << std::endl;
    
    Device device = Device::CPU;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    marker("START");
    
    // Create input
    Tensor input = Tensor::randn<float>(Shape{{256, 1024}}, opts);
    marker("After Input Creation");
    
    // Forward pass - uses checkpointing (does NOT store intermediates)
    std::cout << "Running checkpointed forward pass..." << std::endl;
    variable_list output = checkpoint(deep_computation_checkpointed, {input});
    marker("After Forward - Intermediates NOT STORED");
    
    // Compute loss
    Tensor loss = mean(mul(output[0], output[0]));
    std::cout << "Loss: " << *loss.data<float>() << std::endl;
    marker("After Loss Computation");
    
    // Backward pass - will RECOMPUTE intermediates
    std::cout << "Running backward pass (recomputing intermediates)..." << std::endl;
    loss.backward();
    marker("After Backward - Lower Peak Memory");
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Check Massif - peak should be LOWER than without checkpointing" << std::endl;
    std::cout << "Before cleanup: " << Tensor::get_active_tensor_count() << std::endl;
    
    // Explicitly release all tensors
    input = Tensor();
    output = variable_list();
    loss = Tensor();
    std::cout << "After cleanup: " << Tensor::get_active_tensor_count() << std::endl;

    return 0;
}
