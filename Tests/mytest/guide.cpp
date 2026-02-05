#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "Checkpointing/Checkpoint.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include <iostream>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

int main() {
    // 1. Setup Device
    Device device = Device::CUDA;
    TensorOptions opts = TensorOptions().with_device(device);

    // 2. Define a Simple Model (20 layers)
    auto model = std::make_shared<nn::Sequential>();
    for (int i = 0; i < 20; ++i) {
        model->add(std::make_shared<nn::Linear>(1024, 1024));
        model->add(std::make_shared<nn::ReLU>());
    }
    model->to(device);

    // 3. Setup Optimizer
    auto optimizer = std::make_shared<nn::SGDOptimizer>(model->parameters(), 0.01);

    // 4. Dummy Training Data
    Tensor input = Tensor::randn<float>(Shape{{256, 1024}}, opts.with_req_grad(true));
    Tensor target = Tensor::randn<float>(Shape{{256, 1024}}, opts);

    std::cout << "Starting Training Step..." << std::endl;

    // 5. Training Loop Step
    optimizer->zero_grad();

    // Use checkpoint_sequential to perform memory-efficient forward.
    // This splits the 40 modules (20 layer/relu pairs) into 4 segments.
    // Intermediates WITHIN these segments are freed as soon as layer finishes.
    variable_list outputs = checkpoint_sequential(model, 4, {input});
    Tensor output = outputs[0];

    // Compute Loss
    Tensor diff = sub(output, target);
    Tensor loss = mean(mul(diff, diff));
    std::cout << "Initial Loss: " << loss.to_cpu().data<float>()[0] << std::endl;

    // Backward Pass (Triggers recomputation of segments)
    loss.backward();

    // Update Weights
    optimizer->step();

    std::cout << "Step Complete. Active Tensors: " << Tensor::get_active_tensor_count() << std::endl;

    // 6. Memory Management Note:
    // In C++, once 'input', 'model', and 'loss' go out of scope at the end of main, 
    // the library automatically calls the GPU deallocator. No manual free is required!
    //
    // However, if you are in a long loop and want memory back NOW, you can do:
    // input = Tensor();
    // loss = Tensor();
    
    return 0;
}
