#include "ops/helpers/AdamKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace OwnTensor {
namespace cuda {

__global__ void fused_adam_kernel(
    float* param,
    const float* grad,
    float* m,
    float* v,
    int64_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float g = grad[idx];
    float p = param[idx];
    
    // Apply weight decay (AdamW style - decoupled)
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }
    
    // Update first moment: m = beta1 * m + (1 - beta1) * g
    float m_old = m[idx];
    float m_new = beta1 * m_old + (1.0f - beta1) * g;
    m[idx] = m_new;
    
    // Update second moment: v = beta2 * v + (1 - beta2) * g^2
    float v_old = v[idx];
    float v_new = beta2 * v_old + (1.0f - beta2) * g * g;
    v[idx] = v_new;
    
    // Bias-corrected estimates
    float m_hat = m_new / bias_correction1;
    float v_hat = v_new / bias_correction2;
    
    // Update parameter
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    
    param[idx] = p;
}

void fused_adam_cuda(
    float* param,
    const float* grad,
    float* m,
    float* v,
    int64_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    fused_adam_kernel<<<blocks, threads>>>(
        param, grad, m, v, numel,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2
    );
}

} // namespace cuda
} // namespace OwnTensor
