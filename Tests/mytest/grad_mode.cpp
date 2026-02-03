#include "core/Tensor.h"
#include "Checkpointing/GradMode.h"
#include "autograd/operations/BinaryOps.h"
#include <iostream>
#include <cassert>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

int main() {
    std::cout << "\n=== Testing GradMode (no_grad) ===\n\n";

    TensorOptions opts = TensorOptions{}.with_dtype(Dtype::Float32);
    opts.requires_grad = true;

    Tensor a = Tensor::ones(Shape{{2, 2}}, opts);
    Tensor b = Tensor::ones(Shape{{2, 2}}, opts);

    // Test 1: Default behavior (GradMode enabled)
    std::cout << "Test 1: Default behavior (GradMode enabled)...\n";
    {
        Tensor c = add(a, b);
        assert(c.requires_grad() == true);
        assert(c.grad_fn() != nullptr);
        std::cout << "✓ Graph built as expected\n";
    }

    // Test 2: NoGradGuard
    std::cout << "\nTest 2: NoGradGuard...\n";
    {
        NoGradGuard guard;
        std::cout << "  GradMode::is_enabled() = " << GradMode::is_enabled() << "\n";
        std::cout << "  a.requires_grad() = " << a.requires_grad() << "\n";
        std::cout << "  b.requires_grad() = " << b.requires_grad() << "\n";
        
        Tensor c = add(a, b);
        
        std::cout << "  c.requires_grad() = " << c.requires_grad() << "\n";
        std::cout << "  c.grad_fn() = " << (void*)c.grad_fn().get() << "\n";
        
        assert(c.requires_grad() == false);
        assert(c.grad_fn() == nullptr);
        std::cout << "✓ No graph built inside NoGradGuard\n";
    }

    // Test 3: GradModeGuard (re-enabling)
    std::cout << "\nTest 3: GradModeGuard (re-enabling)...\n";
    {
        NoGradGuard guard;
        {
            GradModeGuard g(true);
            Tensor c = add(a, b);
            assert(c.requires_grad() == true);
            assert(c.grad_fn() != nullptr);
            std::cout << "✓ Graph built inside nested GradModeGuard(true)\n";
        }
        Tensor d = add(a, b);
        assert(d.grad_fn() == nullptr);
        std::cout << "✓ No graph built after exiting nested GradModeGuard\n";
    }

    // Test 4: Nested NoGradGuard
    std::cout << "\nTest 4: Nested NoGradGuard...\n";
    {
        NoGradGuard guard1;
        {
            NoGradGuard guard2;
            Tensor c = add(a, b);
            assert(c.grad_fn() == nullptr);
        }
        std::cout << "✓ Nested NoGradGuard works correctly\n";
    }

    std::cout << "\n=== All GradMode tests PASSED! ===\n\n";
    return 0;
}