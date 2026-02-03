#include "core/Tensor.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "Checkpointing/Checkpoint.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace OwnTensor;
using namespace OwnTensor::nn;

int main() {
    std::cout << "[Checkpoint Training Loop Test] Starting..." << std::endl;

    // 1. Define Model Parameters
    int input_size = 256;
    int hidden_size = 256;
    int num_layers = 10;
    int batch_size = 64;
    int num_epochs = 5; // Run a few epochs to see memory stability

    // 2. Build 10-layer MLP using Sequential
    auto model = std::make_shared<Sequential>(std::initializer_list<Module*>{});
    
    for (int i = 0; i < num_layers; ++i) {
        int in_features = (i == 0) ? input_size : hidden_size;
        int out_features = (i == num_layers - 1) ? 1 : hidden_size; // Final layer outputs 1 value
        
        model->add(std::make_shared<Linear>(in_features, out_features));
        
        if (i < num_layers - 1) {
            model->add(std::make_shared<ReLU>());
        }
    }

    // Move model to GPU if available
    if (Tensor::zeros(Shape{{1}}).is_cuda()) {
        std::cout << "Moving model to CUDA..." << std::endl;
        model->to(DeviceIndex(Device::CUDA, 0));
    }

    // 3. Create Optimizer
    auto params = model->parameters();
    SGDOptimizer optimizer(params, 0.01);

    // 4. Training Loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;

        // Create dummy input and target
        Tensor input = Tensor::randn<float>(Shape{{batch_size, input_size}}, TensorOptions().with_req_grad(true));
        Tensor target = Tensor::randn<float>(Shape{{batch_size, 1}}, TensorOptions());
        
        if (input.is_cuda()) {
            input = input.to_cuda();
            target = target.to_cuda();
        }

        // Forward Pass with Checkpointing
        // We split the 10-layer model into 2 segments for checkpointing
        variable_list inputs = {input};
        variable_list outputs = autograd::checkpoint_sequential(model, 2, inputs);
        Tensor prediction = outputs[0];

        // Compute Loss (MSE)
        Tensor loss = mse_loss(prediction, target);
        // Sync to CPU to read value
        float loss_val = *loss.to_cpu().data<float>();
        std::cout << "  Loss: " << loss_val << std::endl;

        // Backward Pass
        optimizer.zero_grad(); // Use set_to_none=true for maximum memory saving
        loss.backward();

        // Optimizer Step
        optimizer.step();
        
        std::cout << "  Step complete." << std::endl;
    }

    std::cout << "[Checkpoint Training Loop Test] Finished." << std::endl;
    return 0;
}