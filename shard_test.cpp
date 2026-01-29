#include "TensorLib.h"
#include <iostream>

using namespace OwnTensor;

int main()
{
    Tensor A = Tensor::rand({{8, 4 ,2}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0), false}, 42, 0.f, 1.f);
    A.display();
    std::vector<Tensor> shards = A.make_shards(size_t(2), int64_t(2));

    for (Tensor i : shards)
    {
        i.display();
        std::cout << "\n\n";
    }
}
