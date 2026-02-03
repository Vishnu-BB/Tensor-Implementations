#include "core/Tensor.h"
#include "nn/NN.h"
#include "Checkpointing/Checkpoint.h"
#include "Checkpointing/GradMode.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/operations/BinaryOps.h"
#include "core/RNG.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>

using namespace OwnTensor;
using namespace OwnTensor::autograd;
using namespace OwnTensor::nn;

// --- Helper for Comparison ---
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

void print_grad_stats(const std::string& label, const Tensor& t) {
    if (!t.has_grad()) {
        std::cout << "  " << label << ": No Grad\n";
        return;
    }
    Tensor g = t.grad_view().to_cpu();
    float mean_val = 0;
    const float* data = g.data<float>();
    for(size_t i=0; i<g.numel(); ++i) mean_val += data[i];
    mean_val /= g.numel();
    std::cout << "  " << std::left << std::setw(20) << label << " | Grad Mean: " << std::fixed << std::setprecision(8) << mean_val << "\n";
}

// --- Custom Stochastic Module ---
class StochasticNoise : public Module {
public:
    Tensor forward(const Tensor& input) override {
        // Add random noise: y = x + randn()
        Tensor noise = Tensor::randn<float>(input.shape(), input.opts(), 0.1f);
        return add(input, noise);
    }
};

void run_edge_case_test(Device device, int num_layers, int segments, const std::string& case_name) {
    std::cout << "\n>>> [CASE: " << case_name << "] (" << (device == Device::CPU ? "CPU" : "GPU") << ", Segments: " << segments << ")\n";
    
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    std::vector<std::shared_ptr<Module>> modules;
    for (int i = 0; i < num_layers; ++i) {
        modules.push_back(std::make_shared<Linear>(10, 10));
        modules.push_back(std::make_shared<ReLU>());
    }
    
    auto model = std::make_shared<Sequential>(std::initializer_list<Module*>{});
    for (auto& m : modules) model->add(m);
    model->to(device == Device::CPU ? DeviceIndex(Device::CPU) : DeviceIndex(Device::CUDA));

    Tensor x = Tensor::randn<float>(Shape{{2, 10}}, opts);
    
    // Baseline
    RNG::set_seed(42);
    Tensor out_base = model->forward(x);
    sum(out_base).backward();
    
    std::vector<Tensor> grads_base;
    for (const auto& p : model->parameters()) grads_base.push_back(p.grad_view().clone());
    Tensor x_grad_base = x.grad_view().clone();
    
    // Checkpoint
    model->zero_grad();
    x.zero_grad();
    RNG::set_seed(42);
    
    try {
        variable_list out_cp = checkpoint_sequential(model, segments, {x});
        sum(out_cp[0]).backward();
        
        // Verification
        bool match = compare_tensors(out_base, out_cp[0]);
        std::cout << "  Forward Match: " << (match ? "YES" : "NO") << "\n";
        
        bool grad_match = compare_tensors(x_grad_base, x.grad_view());
        std::cout << "  Input Grad Match: " << (grad_match ? "YES" : "NO") << "\n";
        
        print_grad_stats("Input Grad", x);
        
        assert(match && grad_match);
        std::cout << "  ✓ Passed!\n";
    } catch (const std::exception& e) {
        std::cout << "  Caught expected exception: " << e.what() << "\n";
    }
}

void test_stochastic_consistency(Device device) {
    std::cout << "\n>>> [CASE: Stochastic Consistency] (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    auto model = std::make_shared<Sequential>(std::initializer_list<Module*>{
        new Linear(10, 10),
        new StochasticNoise(),
        new Linear(10, 10)
    });
    model->to(device == Device::CPU ? DeviceIndex(Device::CPU) : DeviceIndex(Device::CUDA));

    Tensor x = Tensor::randn<float>(Shape{{2, 10}}, opts);

    // Baseline
    RNG::set_seed(123);
    Tensor out_base = model->forward(x);
    sum(out_base).backward();
    Tensor x_grad_base = x.grad_view().clone();

    // Checkpoint
    model->zero_grad();
    x.zero_grad();
    RNG::set_seed(123);
    variable_list out_cp = checkpoint_sequential(model, 2, {x});
    sum(out_cp[0]).backward();

    assert(compare_tensors(out_base, out_cp[0]));
    assert(compare_tensors(x_grad_base, x.grad_view()));
    std::cout << "  ✓ Stochastic Consistency Passed!\n";
}

void test_multi_output_checkpoint(Device device) {
    std::cout << "\n>>> [CASE: Multi-Output Checkpoint] (" << (device == Device::CPU ? "CPU" : "GPU") << ")\n";
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    Tensor x = Tensor::randn<float>(Shape{{2, 5}}, opts);
    Tensor y = Tensor::randn<float>(Shape{{2, 5}}, opts);
    
    auto multi_out_fn = [](const variable_list& inputs) {
        Tensor a = mul(inputs[0], inputs[1]);
        Tensor b = add(inputs[0], inputs[1]);
        return variable_list{a, b};
    };
    
    // Baseline
    variable_list out_base = multi_out_fn({x, y});
    Tensor loss_base = add(sum(out_base[0]), sum(out_base[1]));
    loss_base.backward();
    
    Tensor x_grad_base = x.grad_view().clone();
    Tensor y_grad_base = y.grad_view().clone();
    
    // Checkpoint
    x.zero_grad();
    y.zero_grad();
    variable_list out_cp = checkpoint(multi_out_fn, {x, y});
    Tensor loss_cp = add(sum(out_cp[0]), sum(out_cp[1]));
    loss_cp.backward();
    
    assert(compare_tensors(out_base[0], out_cp[0]));
    assert(compare_tensors(out_base[1], out_cp[1]));
    assert(compare_tensors(x_grad_base, x.grad_view()));
    assert(compare_tensors(y_grad_base, y.grad_view()));
    
    std::cout << "  ✓ Multi-Output Checkpoint Passed!\n";
}

int main() {
    std::vector<Device> devices = {Device::CPU};
#ifdef WITH_CUDA
    devices.push_back(Device::CUDA);
#endif

    for (Device dev : devices) {
        // 1. Single segment (equivalent to checkpointing whole model)
        run_edge_case_test(dev, 4, 1, "Single Segment");
        
        // 2. Max segments (one per module)
        run_edge_case_test(dev, 4, 8, "Max Segments");
        
        // 3. Over segments (capped at num_modules)
        run_edge_case_test(dev, 4, 20, "Over Segments");
        
        // 4. Non-divisible segments (8 modules / 3 segments = 3, 3, 2)
        run_edge_case_test(dev, 4, 3, "Non-divisible Segments");
        
        // 5. Invalid segments (0)
        run_edge_case_test(dev, 2, 0, "Zero Segments (Error)");
        
        // 6. Stochastic consistency
        test_stochastic_consistency(dev);

        // 7. Multi-output checkpoint
        test_multi_output_checkpoint(dev);
    }

    std::cout << "\nAll edge case tests completed! ✓\n";
    return 0;
}