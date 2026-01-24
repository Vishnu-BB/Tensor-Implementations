#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cmath>

#include "TensorLib.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"

using namespace OwnTensor;

// --- Helper Functions for Binary I/O ---

Tensor load_tensor(const std::string& filename, std::ifstream& f) {
    if (!f.is_open()) throw std::runtime_error("File not open");

    uint32_t num_dims;
    f.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));

    std::vector<int64_t> dims(num_dims);
    size_t num_elements = 1;
    for (uint32_t i = 0; i < num_dims; ++i) {
        uint32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        dims[i] = static_cast<int64_t>(d);
        num_elements *= d;
    }

    std::vector<float> data(num_elements);
    f.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

    // Create tensor on CPU
    Tensor t = Tensor::empty(Shape(dims), {Dtype::Float32, DeviceIndex(Device::CPU, 0)});
    t.set_data(data); 
    return t;
}


void save_tensor(const Tensor& t, std::ofstream& f) {
    // Ensure tensor is on CPU for saving
    Tensor t_cpu = t.to(DeviceIndex(Device::CPU, 0));
    
    // Write dims
    const auto& shape = t_cpu.shape();
    uint32_t num_dims = static_cast<uint32_t>(shape.dims.size());
    f.write(reinterpret_cast<const char*>(&num_dims), sizeof(uint32_t));
    
    for (auto d : shape.dims) {
        uint32_t dim_val = static_cast<uint32_t>(d);
        f.write(reinterpret_cast<const char*>(&dim_val), sizeof(uint32_t));
    }
    
    // Write Data
    const float* ptr = t_cpu.data<float>(); 
    size_t num_elements = t_cpu.numel();
    f.write(reinterpret_cast<const char*>(ptr), num_elements * sizeof(float));
}

int main(int argc, char** argv) {
    (void)argc; (void)argv; // Suppress unused warning
    try {
        std::string data_file = "accuracy_test_data.bin";
        std::string results_file = "accuracy_test_results.bin";
        
        // Open Data File
        std::ifstream fin(data_file, std::ios::binary);
        if (!fin) throw std::runtime_error("Could not open data file: " + data_file);
        
        // Load Inputs in Order
        Tensor input = load_tensor(data_file, fin);
        Tensor target = load_tensor(data_file, fin);
        
        Tensor fc1_w_init = load_tensor(data_file, fin);
        Tensor fc1_b_init = load_tensor(data_file, fin);
        
        Tensor fc2_w_init = load_tensor(data_file, fin);
        Tensor fc2_b_init = load_tensor(data_file, fin);
        
        Tensor fc3_w_init = load_tensor(data_file, fin);
        Tensor fc3_b_init = load_tensor(data_file, fin);
        
        fin.close();
        
        // Setup Device
        Device dev_type = Device::CPU; // Default
        const char* env_dev = std::getenv("TEST_DEVICE");
        if (env_dev) {
            std::string s(env_dev);
            if (s == "CUDA" || s == "cuda") {
                if (device::cuda_available()) {
                    dev_type = Device::CUDA;
                    std::cout << "[C++] Using Device: CUDA" << std::endl;
                } else {
                    std::cerr << "[C++] CUDA requested but not available. Fallback to CPU." << std::endl;
                }
            } else {
                 std::cout << "[C++] Using Device: CPU" << std::endl;
            }
        } else {
             std::cout << "[C++] Using Device: CPU (default)" << std::endl;
        }

        DeviceIndex dev_idx(dev_type, 0);
        
        // Move to Device
        input = input.to(dev_idx);
        target = target.to(dev_idx);
        // input.set_requires_grad(true); // Don't typically need grad for input
        
        // Prepare Weights
        // We use loaded weights.
        Tensor w1 = fc1_w_init.to(dev_idx); w1.set_requires_grad(true);
        Tensor b1 = fc1_b_init.to(dev_idx); b1.set_requires_grad(true);
        
        Tensor w2 = fc2_w_init.to(dev_idx); w2.set_requires_grad(true);
        Tensor b2 = fc2_b_init.to(dev_idx); b2.set_requires_grad(true);
        
        Tensor w3 = fc3_w_init.to(dev_idx); w3.set_requires_grad(true);
        Tensor b3 = fc3_b_init.to(dev_idx); b3.set_requires_grad(true);
        
        // --- Phase 1: Accuracy Check (1 Iteration) ---
        // We run exactly one pass to match the python script's "one step" for precise value comparison.
        
        // Forward Pass
        // Layer 1: Linear (x @ w.T + b)
        Tensor x1 = autograd::matmul(input, w1.transpose(0, 1)) + b1;
        Tensor relu_out = autograd::relu(x1);
        
        // Layer 2
        Tensor x2 = autograd::matmul(relu_out, w2.transpose(0, 1)) + b2;
        Tensor sigmoid_out = autograd::sigmoid(x2);
        
        // Layer 3
        Tensor final_out = autograd::matmul(sigmoid_out, w3.transpose(0, 1)) + b3;
        
        // Loss
        Tensor diff = final_out - target;
        Tensor diff_sq = diff * diff;
        Tensor loss = autograd::mean(diff_sq);
        
        // Backward
        loss.backward();
        
        std::cout << "[C++] Backward pass completed." << std::endl;

        // Optimizer Step & Save for Accuracy Check
        Tensor w1_grad_check, w1_new_check;
        if (w1.has_grad()) {
            w1_grad_check = w1.grad_view(); // Save grad for check
            // We do NOT apply update in-place for the weight-check variable because we might want to continue using w1?
            // Actually, for the check we want the *updated weight*.
            // Let's compute it.
            w1_new_check = w1 - w1_grad_check * 0.01f;
        } else {
             w1_grad_check = Tensor::zeros(w1.shape(), w1.opts());
             w1_new_check = w1; 
        }

        // Save Results (Accuracy Phase)
        std::ofstream fout(results_file, std::ios::binary);
        if (!fout) throw std::runtime_error("Could not open results file");
        
        save_tensor(relu_out, fout);
        save_tensor(sigmoid_out, fout);
        save_tensor(final_out, fout);
        save_tensor(loss, fout);
        save_tensor(w1_grad_check, fout);      
        save_tensor(w1_new_check, fout);       
        
        // --- Phase 2: Throughput Benchmark (N Iterations) ---
        int iterations = 100;
        int warmup_iters = 10;
        
        // Ensure gradients are cleared
        w1.zero_grad(); b1.zero_grad();
        w2.zero_grad(); b2.zero_grad();
        w3.zero_grad(); b3.zero_grad();
        
        auto update_param = [](Tensor& p, float lr) {
            if (p.has_grad()) {
                Tensor g = p.grad_view();
                p = (p - g * lr);
                p.set_grad_fn(nullptr);
                p.set_requires_grad(true);
            }
        };

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            Tensor x1_ = autograd::matmul(input, w1.transpose(0, 1)) + b1;
            Tensor relu_ = autograd::relu(x1_);
            Tensor x2_ = autograd::matmul(relu_, w2.transpose(0, 1)) + b2;
            Tensor sig_ = autograd::sigmoid(x2_);
            Tensor final_ = autograd::matmul(sig_, w3.transpose(0, 1)) + b3;
            Tensor diff_ = final_ - target;
            Tensor loss_ = autograd::mean(diff_ * diff_);
            loss_.backward();
            
            update_param(w1, 0.01f); update_param(b1, 0.01f);
            update_param(w2, 0.01f); update_param(b2, 0.01f);
            update_param(w3, 0.01f); update_param(b3, 0.01f);
        }

        if (dev_type == Device::CUDA) {
            #ifdef WITH_CUDA
            cudaDeviceSynchronize();
            #endif
        }

        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            Tensor x1_ = autograd::matmul(input, w1.transpose(0, 1)) + b1;
            Tensor relu_ = autograd::relu(x1_);
            Tensor x2_ = autograd::matmul(relu_, w2.transpose(0, 1)) + b2;
            Tensor sig_ = autograd::sigmoid(x2_);
            Tensor final_ = autograd::matmul(sig_, w3.transpose(0, 1)) + b3;
            Tensor diff_ = final_ - target;
            Tensor loss_ = autograd::mean(diff_ * diff_);
            loss_.backward();
            
            update_param(w1, 0.01f); update_param(b1, 0.01f);
            update_param(w2, 0.01f); update_param(b2, 0.01f);
            update_param(w3, 0.01f); update_param(b3, 0.01f);
        }

        if (dev_type == Device::CUDA) {
            #ifdef WITH_CUDA
            cudaDeviceSynchronize();
            #endif
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        float avg_ms = duration.count() / iterations;
        
        std::cout << "Benchmark (N=" << iterations << ") Average Time: " << avg_ms << " ms" << std::endl;
        
        // Write Average Throughput
        float ms_total = duration.count();
        float ms_avg = ms_total / iterations;
        fout.write(reinterpret_cast<const char*>(&ms_avg), sizeof(float));
        
        if (!fout) std::cerr << "Error writing throughput!" << std::endl;
        
        fout.close();
        std::cout << "[C++] Results saved successfully." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "C++ Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
