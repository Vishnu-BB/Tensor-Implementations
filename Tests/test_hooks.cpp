#include <iostream>
#include <cassert>
#include "core/Tensor.h"
#include "autograd/Variable.h"
#include "autograd/Hooks.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"

using namespace OwnTensor;

bool hook_called = false;

void test_post_acc_hook() {
    std::cout << "Testing PostAccumulateGradHook..." << std::endl;
    
    // 1. Create a leaf parameter
    Tensor x = Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    
    // 2. Register a post-accumulation hook
    x.register_post_acc_hook(std::make_unique<LambdaPostAccHook>(
        [](const Tensor& grad) {
            std::cout << "Hook called!" << std::endl;
            hook_called = true;
        }
    ));
    
    // 3. Simple operation: y = x + x
    Tensor y = autograd::add(x, x);
    
    // 4. Sum to scalar for backward
    Tensor loss = autograd::sum(y);
    
    // 5. Backward
    loss.backward();
    
    // 6. Verify hook was called
    if (hook_called) {
        std::cout << "SUCCESS: Hook was called!" << std::endl;
    } else {
        std::cout << "FAILURE: Hook was NOT called!" << std::endl;
        exit(1);
    }
}

int main() {
    try {
        test_post_acc_hook();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
