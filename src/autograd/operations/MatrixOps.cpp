#include "autograd/operations/MatrixOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/MatrixBackward.h"
#include "ops/Kernels.h"
#include <algorithm>

namespace OwnTensor {
namespace autograd {

Tensor matmul(const Tensor& a, const Tensor& b) {
    return make_binary_op<MatmulBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return OwnTensor::matmul(x, y); },
        a, b);  // Pass a, b to MatmulBackward constructor
}



} // namespace autograd
} // namespace OwnTensor