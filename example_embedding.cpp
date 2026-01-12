#include <iostream>
#include <vector>
#include <cassert>
#include "nn/NN.h"
#include "autograd/AutogradOps.h"
#include "core/Tensor.h"

using namespace OwnTensor;
using namespace OwnTensor::nn;

/**
 * Example demonstrating the Embedding layer with Shape [B, T] -> [B, T, C]
 * and automatic gradient accumulation (scatter-add).
 */
int main() {
    try {
        std::cout << "--- OwnTensor Embedding Example ---\n";

        // 1. Setup Configuration
        const int vocab_size = 50257; // GPT-2 vocab size
        const int embed_dim = 128;
        const int batch_size = 2;
        const int seq_len = 4;
        const int padding_idx = 0;

        // 2. Initialize Layer
        Embedding emb(vocab_size, embed_dim, padding_idx);
        
        // Move to GPU if available (optional)
        // if (device::cuda_available()) {
        //     emb.weight = emb.weight.to(DeviceIndex(Device::CUDA));
        // }

        // 3. Create Input [B, T]
        // Using uint16_t for token IDs
        Tensor indices(Shape{{batch_size, seq_len}}, Dtype::UInt16);
        std::vector<uint16_t> tokens = {
            10, 20, 30, 0,  // Batch 1 (one pad)
            40, 50, 0,  0   // Batch 2 (two pads)
        };
        indices.set_data(tokens);

        std::cout << "Input shape: " << batch_size << "x" << seq_len << "\n";

        // 4. Forward Pass [B, T] -> [B, T, C]
        Tensor output = emb.forward(indices);
        
        std::cout << "Output shape: ";
        for(auto d : output.shape().dims) std::cout << d << " ";
        std::cout << "(Expected: " << batch_size << " " << seq_len << " " << embed_dim << ")\n";
        
        assert(output.shape().dims.size() == 3);
        assert(output.shape().dims[2] == embed_dim);

        // 5. Backward Pass
        // Assume some loss (e.g., mean)
        Tensor loss = autograd::mean(output);
        std::cout << "Initial Loss: " << loss.to_cpu().data<float>()[0] << "\n";
        
        loss.backward();

        // 6. Verify Gradients [V, C]
        assert(emb.weight.grad() != nullptr);
        std::cout << "Gradients computed for weight matrix of shape: ";
        for(auto d : emb.weight.shape().dims) std::cout << d << " ";
        std::cout << "\n";

        // Check padding_idx gradient is zero
        Tensor grad_cpu = emb.weight.grad_view().to_cpu();
        float* g_ptr = grad_cpu.data<float>();
        bool pad_grad_zero = true;
        for(int j=0; j<embed_dim; ++j) {
            if (g_ptr[padding_idx * embed_dim + j] != 0.0f) {
                pad_grad_zero = false;
                break;
            }
        }
        
        if (pad_grad_zero) {
            std::cout << "SUCCESS: Padding index gradients are zero.\n";
        } else {
            std::cout << "FAILURE: Padding index has non-zero gradients.\n";
        }

        std::cout << "✅ Example completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}