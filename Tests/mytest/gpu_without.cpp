#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "nn/NN.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace OwnTensor;
using namespace OwnTensor::autograd;


// Helper function to check and report GPU memory usage
void check_gpu_memory(const std::string& step_name) {
    size_t free_bytes, total_bytes;
    // Get memory info for the current device
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (status == cudaSuccess) {
        size_t used_bytes = total_bytes - free_bytes;
        std::cout << step_name << std::endl;
        std::cout << "  Total Memory (MB): " << total_bytes / (1024.0 * 1024.0) << std::endl;
        std::cout << "  Free Memory (MB):  " << free_bytes / (1024.0 * 1024.0) << std::endl;
        std::cout << "  Used Memory (MB):  " << used_bytes / (1024.0 * 1024.0) << std::endl;
    } else {
        std::cerr << "Error getting GPU memory info: " << cudaGetErrorString(status) << std::endl;
    }
}

void marker(const std::string& label) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "Active Tensors: " << Tensor::get_active_tensor_count() << std::endl;
}

int main() {
    std::cout << "=== GPU WITHOUT Checkpointing (Baseline) ===" << std::endl;

#ifndef WITH_CUDA
    std::cout << "SKIPPING: CUDA not enabled." << std::endl;
    return 0;
#endif

    Device device = Device::CUDA;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);

    marker("START");
    check_gpu_memory("After Marker START");

    // 1. Setup Data
    Tensor input = Tensor::randn<float>(Shape{{256, 1024}}, opts);
    Tensor target = Tensor::randn<float>(Shape{{256, 1024}}, opts.with_req_grad(false));

    // 2. Setup Deep MLP (20 layers)
    auto model = std::make_shared<nn::Sequential>();
    for (int i = 0; i < 20; ++i) {
        model->add(std::make_shared<nn::Linear>(1024, 1024));
        model->add(std::make_shared<nn::ReLU>());
    }
    model->to(device);

    marker("After Model & Input Creation");
    check_gpu_memory("After Marker After Model & Input Creation");

    // 3. Forward Pass (storing ALL intermediates)
    std::cout << "Running forward pass..." << std::endl;
    Tensor output = input;
    for (const auto& module : model->modules()) {
        output = module->forward(output);
    }
    check_gpu_memory("After Forward - Intermediates STORED in GPU memory");    
    marker("After Forward - Intermediates STORED in GPU memory");

    // 4. Loss & Backward
    Tensor diff = sub(output, target);
    Tensor loss = mean(mul(diff, diff));
    std::cout << "Loss: " << loss.to_cpu().data<float>()[0] << std::endl;
    check_gpu_memory("After Loss Computation");    
    marker("After Loss Computation");

    std::cout << "Running backward pass..." << std::endl;
    loss.backward();
    check_gpu_memory("After Backward - Peak Memory Reached");    
    marker("After Backward - Peak Memory Reached");

    // 5. Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    input = Tensor();
    target = Tensor();
    output = Tensor();
    loss = Tensor();
    model = nullptr;

    check_gpu_memory("After Cleanup - Should be close to 0");    
    marker("After Cleanup - Should be close to 0");

    return 0;
}
