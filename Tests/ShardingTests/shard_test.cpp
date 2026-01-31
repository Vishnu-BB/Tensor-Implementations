#include "TensorLib.h"
#include <iostream>

using namespace OwnTensor;

int main()
{
    Tensor A = Tensor::rand({{2, 4 ,2}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0), false}, 42, 0.f, 1.f);
    std::cout << "Address: " << A.data() << std::endl;
    A.display();
    // std::vector<Tensor> shards = A.make_shards_inplace(size_t(2), int64_t(2));

    // for (Tensor i : shards)
    // {
    //     std::cout << "Address: " << i.data() << std::endl;
    //     i.display();
    //     std::cout << "\n\n";
    // }
    Tensor B = A.slice(0, 5);
    std::cout << "\nAddress: " << B.data() << std::endl;
    B.display();
    assert(B.data() != (A.data() + 0*4));

    Tensor C = A.slice_inplace(0, 5);
    std::cout << "\nAddress: " << C.data() << std::endl;
    C.display();
    assert(C.data() == (A.data() + 0*4));
}
