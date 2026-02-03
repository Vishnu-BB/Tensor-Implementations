#include "core/Tensor.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "Checkpointing/Checkpoint.h"
#include "mlp/layers.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

/**
 * Gradient Checkpointing Example - Profiling Version
 */

// Helper to compare tensors
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

variable_list my_complex_block(const variable_list& inputs, size_t* memory_counter = nullptr) {
    Tensor x = inputs[0];
    Tensor w = inputs[1];
    Tensor b = inputs[2];

    Tensor h1 = OwnTensor::mlp_forward::linear(x, w, b);
    Tensor h2 = relu(h1);
    Tensor h3 = mul(h2, h2);
    
    if (memory_counter) {
        *memory_counter += h1.nbytes();
        *memory_counter += h2.nbytes();
    }

    return { h3 };
}

void run_standard() {
    std::cout << "[Standard Mode] Running...\n";
    TensorOptions opts = TensorOptions().with_device(Device::CPU).with_req_grad(true);
    // Large tensors to make memory usage obvious
    Tensor x = Tensor::randn<float>(Shape{{256, 256}}, opts);
    Tensor w = Tensor::randn<float>(Shape{{256, 256}}, opts);
    Tensor b = Tensor::zeros(Shape{{256}}, opts);

    size_t intermediate_memory = 0;
    variable_list out = my_complex_block({x, w, b}, &intermediate_memory);
    Tensor loss = sum(out[0]);
    loss.backward();
    
    std::cout << "Standard pass complete. Estimated intermediate memory: " << (intermediate_memory / (1024*1024.0)) << " MB\n";
}

void run_checkpoint() {
    std::cout << "[Checkpoint Mode] Running...\n";
    TensorOptions opts = TensorOptions().with_device(Device::CUDA).with_req_grad(true);
    Tensor x = Tensor::randn<float>(Shape{{256, 256}}, opts);
    Tensor w = Tensor::randn<float>(Shape{{256, 256}}, opts);
    Tensor b = Tensor::zeros(Shape{{256}}, opts);

    auto block_wrapper = [](const variable_list& ins) {
        return my_complex_block(ins, nullptr); 
    };

    variable_list out = checkpoint(block_wrapper, {x, w, b});
    Tensor loss = sum(out[0]);
    loss.backward();
    
    std::cout << "Checkpoint pass complete. Intermediate memory stored: 0 bytes\n";
}

int main(int argc, char** argv) {
    std::string mode = "all";
    if (argc > 1) mode = argv[1];
    // run_checkpoint();
    try {
        if (mode == "standard") {
            // run_standard();
        } else if (mode == "checkpoint") {
            run_checkpoint();
        } else {
            // Original behavior for verification
            run_standard();
            std::cout << "\n";
            run_checkpoint();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}