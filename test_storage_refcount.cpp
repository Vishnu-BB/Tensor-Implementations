#include <iostream>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void test_simple_detach() {
    std::cout << "Testing simple detach..." << std::endl;
    // creating scope
    Tensor b;
    {
        Tensor a = Tensor::ones(Shape{{10}});
        a.fill(1.0f);
        
        // b shares storage with a
        b = a.detach();
        
        // Modify a, b should see it (shared data)
        a.data<float>()[0] = 5.0f;
        assert(b.data<float>()[0] == 5.0f);
        
        std::cout << "  Shared data verified." << std::endl;
    } 
    // a is destroyed. b should still be valid.
    
    // Verify b is still valid and data is there
    assert(b.numel() == 10);
    assert(b.data<float>()[0] == 5.0f);
    
    // Modify b
    b.data<float>()[1] = 9.0f;
    assert(b.data<float>()[1] == 9.0f);
    
    std::cout << "  Lifetime verified." << std::endl;
}

int main() {
    try {
        test_simple_detach();
        std::cout << "ALL TESTS PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
