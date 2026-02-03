#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "Checkpointing/Checkpoint.h"
#include "Checkpointing/GradMode.h"
#include "core/RNG.h"
#include "mlp/layers.h"
#include "ops/TensorOps.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>

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

// 1. Unused Input Test
void test_unused_input(Device device) {
    std::cout << "  - Running test_unused_input..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);
    Tensor y = Tensor::full(Shape{{2, 2}}, opts, 3.0f); // y is unused but requires grad

    auto fn = [](const variable_list& inputs) {
        return variable_list{ add(inputs[0], inputs[0]) };
    };

    variable_list outs = checkpoint(fn, {x, y});
    Tensor loss = sum(outs[0]);
    loss.backward();

    // dL/dx = 2, dL/dy = 0
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 2.0f)));
    // y's grad should not be modified (it stays 0 if initialized to 0)
    // In our system, if no path exists, grad remains what it was or is null.
    // If it's a leaf and requires grad, it might have a GradAccumulator but no grads sent to it.
}

// 2. Multiple Outputs Test
void test_multiple_outputs(Device device) {
    std::cout << "  - Running test_multiple_outputs..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);
    Tensor y = Tensor::full(Shape{{2, 2}}, opts, 3.0f);

    auto fn = [](const variable_list& inputs) {
        return variable_list{ add(inputs[0], inputs[1]), mul(inputs[0], inputs[1]) };
    };

    variable_list outs = checkpoint(fn, {x, y});
    Tensor loss = add(sum(outs[0]), sum(outs[1]));
    loss.backward();

    // dL/dx = 1 + y = 1 + 3 = 4
    // dL/dy = 1 + x = 1 + 2 = 3
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 4.0f)));
    assert(compare_tensors(y.grad_view(), Tensor::full(y.shape(), opts, 3.0f)));
}

// 3. No Grad Input Test
void test_no_grad_input(Device device) {
    std::cout << "  - Running test_no_grad_input..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(false);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);
    
    auto fn = [](const variable_list& inputs) {
        return variable_list{ mul(inputs[0], inputs[0]) };
    };

    // Even if inputs don't require grad, checkpoint should work (but maybe skip building CheckpointNode)
    variable_list outs = checkpoint(fn, {x});
    assert(outs[0].to_cpu().data<float>()[0] == 4.0f);
    assert(!outs[0].requires_grad());
}

// 4. Mixed Grad Input Test
void test_mixed_grad(Device device) {
    std::cout << "  - Running test_mixed_grad..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts.with_req_grad(true), 2.0f);
    Tensor y = Tensor::full(Shape{{2, 2}}, opts.with_req_grad(false), 3.0f);

    auto fn = [](const variable_list& inputs) {
        return variable_list{ mul(inputs[0], inputs[1]) };
    };

    variable_list outs = checkpoint(fn, {x, y});
    Tensor loss = sum(outs[0]);
    loss.backward();

    // dL/dx = y = 3
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), x.opts(), 3.0f)));
}

// 5. RNG Consistency Test
void test_rng_consistency(Device device) {
    std::cout << "  - Running test_rng_consistency..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{1024, 1024}}, opts);

    auto stochastic_fn = [](const variable_list& inputs) {
        // Dropout-like stochasticity
        Tensor x = inputs[0];
        Tensor mask = Tensor::rand(x.shape(), x.opts(), 0, 0.0f, 1.0f);
        Tensor cond = (mask > Tensor::full(mask.shape(), x.opts(), 0.5f)).as_type(Dtype::Int32);
        return variable_list{ mul(x, cond) };
    };

    // Baseline (no checkpoint)
    RNG::set_seed(1234);
    Tensor x_base = x.clone();
    variable_list outs_base = stochastic_fn({x_base});
    Tensor loss_base = sum(outs_base[0]);
    loss_base.backward();

    // Checkpoint
    RNG::set_seed(1234);
    Tensor x_cp = x.clone();
    variable_list outs_cp = checkpoint(stochastic_fn, {x_cp});
    Tensor loss_cp = sum(outs_cp[0]);
    loss_cp.backward();

    assert(compare_tensors(outs_base[0], outs_cp[0]));
    assert(compare_tensors(x_base.grad_view(), x_cp.grad_view()));
}

// 6. Nested Checkpointing Test
variable_list level2_block(const variable_list& inputs) {
    return { mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f)) };
}

variable_list level1_block(const variable_list& inputs) {
    variable_list inner = checkpoint(level2_block, {inputs[0]});
    return { add(inner[0], inputs[1]) };
}

void test_nested_checkpointing(Device device) {
    std::cout << "  - Running test_nested_checkpointing..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{2, 2}}, opts);
    Tensor y = Tensor::ones(Shape{{2, 2}}, opts);

    variable_list outs = checkpoint(level1_block, {x, y});
    Tensor loss = sum(outs[0]);
    loss.backward();

    // out = 2x + y
    // dL/dx = 2, dL/dy = 1
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 2.0f)));
    assert(compare_tensors(y.grad_view(), Tensor::full(y.shape(), opts, 1.0f)));
}

// 7. Divergent Paths Test
void test_divergent_paths(Device device) {
    std::cout << "  - Running test_divergent_paths..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 4.0f);
    Tensor y = Tensor::full(Shape{{2, 2}}, opts, 5.0f);

    auto fn = [](const variable_list& inputs) {
        Tensor x = inputs[0];
        Tensor y = inputs[1];
        Tensor z = mul(x, y);
        return variable_list{ add(z, x), sub(z, y) };
    };

    variable_list outs = checkpoint(fn, {x, y});
    Tensor loss = add(sum(outs[0]), sum(outs[1]));
    loss.backward();

    // out1 = x*y + x, out2 = x*y - y
    // loss = sum(2xy + x - y)
    // dL/dx = 2y + 1 = 11
    // dL/dy = 2x - 1 = 7
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 11.0f)));
    assert(compare_tensors(y.grad_view(), Tensor::full(y.shape(), opts, 7.0f)));
}

// 8. Chained Checkpoints Test
void test_chained_checkpoints(Device device) {
    std::cout << "  - Running test_chained_checkpoints..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);

    auto f1 = [](const variable_list& inputs) { return variable_list{ mul(inputs[0], inputs[0]) }; }; // x^2
    auto f2 = [](const variable_list& inputs) { return variable_list{ add(inputs[0], inputs[0]) }; }; // 2*x^2

    variable_list o1 = checkpoint(f1, {x});
    variable_list o2 = checkpoint(f2, {o1[0]});
    Tensor loss = sum(o2[0]);
    loss.backward();

    // dL/dx = d(2x^2)/dx = 4x = 8
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 8.0f)));
}

// 9. Shared Inputs Test
void test_shared_inputs(Device device) {
    std::cout << "  - Running test_shared_inputs..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);

    auto fn = [](const variable_list& inputs) {
        // inputs[0] and inputs[1] are the same tensor 'x'
        return variable_list{ mul(inputs[0], inputs[1]) }; // x*x
    };

    variable_list outs = checkpoint(fn, {x, x});
    Tensor loss = sum(outs[0]);
    loss.backward();

    // dL/dx = 2x = 4
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 4.0f)));
}

// 10. Complex Graph Test (Diamond)
void test_complex_graph(Device device) {
    std::cout << "  - Running test_complex_graph..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::full(Shape{{2, 2}}, opts, 2.0f);

    auto fn = [](const variable_list& inputs) {
        Tensor x = inputs[0];
        Tensor a = mul(x, x); // x^2
        Tensor b = add(x, x); // 2x
        return variable_list{ add(a, b) }; // x^2 + 2x
    };

    variable_list outs = checkpoint(fn, {x});
    Tensor loss = sum(outs[0]);
    loss.backward();

    // dL/dx = 2x + 2 = 6
    assert(compare_tensors(x.grad_view(), Tensor::full(x.shape(), opts, 6.0f)));
}

// 11. Parameter Update Test
void test_parameter_updates(Device device) {
    std::cout << "  - Running test_parameter_updates..." << std::endl;
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    Tensor weight = Tensor::randn<float>(Shape{{10, 10}}, opts);
    Tensor bias = Tensor::zeros(Shape{{10}}, opts);
    Tensor input = Tensor::randn<float>(Shape{{5, 10}}, opts.with_req_grad(false));
    Tensor target = Tensor::ones(Shape{{5, 10}}, opts.with_req_grad(false));

    auto linear_block = [&](const variable_list& inputs) {
        // Correct way to use autograd aware ops
        Tensor w_t = inputs[1].t();
        Tensor out = OwnTensor::autograd::matmul(inputs[0], w_t);
        return variable_list{ OwnTensor::autograd::add(out, inputs[2]) };
    };

    for (int i = 0; i < 2; ++i) {
        variable_list out = checkpoint(linear_block, {input, weight, bias});
        Tensor diff = sub(out[0], target);
        Tensor loss = sum(mul(diff, diff));
        
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
    // If it reaches here without crash, it's mostly correct.
}

int main() {
    std::vector<Device> devices = {Device::CPU};
#ifdef WITH_CUDA
    devices.push_back(Device::CUDA);
#endif

    for (Device dev : devices) {
        std::cout << "\n=== Testing Edge Cases on " << (dev == Device::CPU ? "CPU" : "GPU") << " ===" << std::endl;
        try {
            test_unused_input(dev);
            test_multiple_outputs(dev); // Identical to test_multiple_outputs in the user's prompt
            test_no_grad_input(dev);
            test_mixed_grad(dev);
            test_rng_consistency(dev);
            test_nested_checkpointing(dev); // Identical to test_nested_checkpoint
            test_divergent_paths(dev);
            test_chained_checkpoints(dev);
            test_shared_inputs(dev);
            test_complex_graph(dev);
            test_parameter_updates(dev);
        } catch (const std::exception& e) {
            std::cerr << "FAILED: " << e.what() << std::endl;
            return 1;
        }
    }

    std::cout << "\nAll GPU/CPU edge case tests passed! ✓" << std::endl;
    return 0;
}
