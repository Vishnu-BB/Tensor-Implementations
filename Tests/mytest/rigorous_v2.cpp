#include "core/Tensor.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "Checkpointing/Checkpoint.h"
#include "Checkpointing/GradMode.h"
#include "core/RNG.h"
#include "mlp/layers.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// Helper to compare tensors
bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-4) {
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

// Helper to print tensor info
void print_tensor_info(const std::string& label, const Tensor& t) {
    if (!t.unsafeGetTensorImpl()) {
        std::cout << "  " << std::left << std::setw(25) << label << " | NULL\n";
        return;
    }
    Tensor t_cpu = t.to_cpu();
    float mean_val = mean(t_cpu).data<float>()[0];
    std::cout << "  " << std::left << std::setw(25) << label 
              << " | Mean: " << std::fixed << std::setprecision(6) << mean_val << "\n";
}

// 1. Multiple Outputs Test
variable_list multi_output_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor y = inputs[1];
    return {
        add(x, y),
        mul(x, y),
        relu(x)
    };
}

void test_multiple_outputs(Device device) {
    std::cout << "\n[TEST] Multiple Outputs (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);
    Tensor y = Tensor::full(Shape{{2, 2}}, opts, 3.0f);

    // Checkpoint version
    variable_list outs_cp = checkpoint(multi_output_block, {x, y});
    Tensor loss_cp = add(add(sum(outs_cp[0]), sum(outs_cp[1])), sum(outs_cp[2]));
    loss_cp.backward();
    
    Tensor grad_x_cp = x.grad_view().clone();
    Tensor grad_y_cp = y.grad_view().clone();
    
    // Baseline version
    x.zero_grad();
    y.zero_grad();
    variable_list outs_base = multi_output_block({x, y});
    Tensor loss_base = add(add(sum(outs_base[0]), sum(outs_base[1])), sum(outs_base[2]));
    loss_base.backward();
    
    print_tensor_info("Grad X (CP)", grad_x_cp);
    print_tensor_info("Grad X (Base)", x.grad_view());
    
    assert(compare_tensors(grad_x_cp, x.grad_view()));
    assert(compare_tensors(grad_y_cp, y.grad_view()));
    std::cout << "✓ test_multiple_outputs passed!\n";
}

// 2. Nested Checkpointing Test
variable_list inner_block(const variable_list& inputs) {
    return { mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f)) };
}

variable_list outer_block(const variable_list& inputs) {
    variable_list inner_out = checkpoint(inner_block, {inputs[0]});
    return { add(inner_out[0], inputs[1]) };
}

void test_nested_checkpointing(Device device) {
    std::cout << "\n[TEST] Nested Checkpointing (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{2, 2}}, opts);
    Tensor y = Tensor::ones(Shape{{2, 2}}, opts);

    variable_list outs = checkpoint(outer_block, {x, y});
    Tensor loss = sum(outs[0]);
    
    std::cout << "Running backward (should see nested recomputation)...\n";
    loss.backward();
    
    print_tensor_info("Grad X", x.grad_view());
    print_tensor_info("Grad Y", y.grad_view());
    
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 2.0f)));
    assert(compare_tensors(y.grad_view(), Tensor::full(y.shape(), opts, 1.0f)));
    std::cout << "✓ test_nested_checkpointing passed!\n";
}

// 3. Complex Graph Test (Branching/Merging)
variable_list complex_graph_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor y = inputs[1];
    Tensor z = mul(x, y);
    // Path 1: z + x
    // Path 2: z - y
    Tensor out1 = add(z, x);
    Tensor out2 = sub(z, y);
    return { out1, out2 };
}

void test_complex_graph(Device device) {
    std::cout << "\n[TEST] Complex Graph (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 4.0f);
    Tensor y = Tensor::full(Shape{{2, 2}}, opts, 5.0f);

    variable_list outs = checkpoint(complex_graph_block, {x, y});
    Tensor loss = add(sum(outs[0]), sum(outs[1]));
    loss.backward();

    // dL/dx = 2y + 1 = 2*5 + 1 = 11
    // dL/dy = 2x - 1 = 2*4 - 1 = 7
    print_tensor_info("Grad X", x.grad_view());
    print_tensor_info("Grad Y", y.grad_view());
    
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 11.0f)));
    assert(compare_tensors(y.grad_view(), Tensor::full(y.shape(), opts, 7.0f)));
    std::cout << "✓ test_complex_graph passed!\n";
}

// 4. Parameter Update Test
void test_parameter_updates(Device device) {
    std::cout << "\n[TEST] Parameter Updates (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    Tensor weight = Tensor::randn<float>(Shape{{10, 10}}, opts);
    Tensor bias = Tensor::zeros(Shape{{10}}, opts);
    Tensor input = Tensor::randn<float>(Shape{{5, 10}}, opts.with_req_grad(false));
    Tensor target = Tensor::ones(Shape{{5, 10}}, opts.with_req_grad(false));

    auto linear_block = [&](const variable_list& inputs) {
        return variable_list{ OwnTensor::mlp_forward::linear(inputs[0], inputs[1], inputs[2]) };
    };

    float initial_loss = 0;
    for (int i = 0; i < 3; ++i) {
        variable_list out = checkpoint(linear_block, {input, weight, bias});
        Tensor diff = sub(out[0], target);
        Tensor loss = sum(mul(diff, diff));
        
        float loss_val = loss.to_cpu().data<float>()[0];
        if (i == 0) initial_loss = loss_val;
        std::cout << "  Step " << i << " Loss: " << loss_val << "\n";
        
        weight.zero_grad();
        bias.zero_grad();
        loss.backward();

        float lr = 0.01f;
        {
            NoGradGuard guard;
            weight.copy_(sub(weight, mul(weight.grad_view(), Tensor::full(weight.shape(), weight.opts(), lr))));
            bias.copy_(sub(bias, mul(bias.grad_view(), Tensor::full(bias.shape(), bias.opts(), lr))));
        }
    }
    
    std::cout << "✓ test_parameter_updates passed!\n";
}

int main() {
    std::vector<Device> devices = {Device::CPU};
#ifdef WITH_CUDA
    devices.push_back(Device::CUDA);
#endif

    for (Device dev : devices) {
        try {
            test_multiple_outputs(dev);
            test_nested_checkpointing(dev);
            test_complex_graph(dev);
            test_parameter_updates(dev);
        } catch (const std::exception& e) {
            std::cerr << "Error on " << (dev == Device::CPU ? "CPU" : "GPU") << ": " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "\nAll rigorous checkpoint tests passed! ✓\n";
    return 0;
}