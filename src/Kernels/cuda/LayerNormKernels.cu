#include "ops/helpers/LayerNormKernels.h"
// Unused headers removed

namespace OwnTensor {
namespace cuda {

// =================================================================================
// Helper: Warp Reduction
// =================================================================================
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =================================================================================
// Forward Kernel
// =================================================================================
// Grid: [rows], Block: [min(cols, 1024)] (or fixed size with loop)
// We'll use 1 block per row.
__global__ void layer_norm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int cols,
    float eps) 
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Offset to current row
    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // 1. Mean
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += row_x[i];
    }
    sum = warpReduceSum(sum);
    // Block reduction
    __shared__ float shared_mean;
    if (tid % warpSize == 0) atomicAdd(&shared_mean, sum); // Simple atomic for inter-warp (assuming few warps)
    // For strictly correct block reduce we need shared mem scan, 
    // but atomic is fine for typical hidden sizes (768-4096) with few warps.
    // Actually, properly zero init shared mem.
    if (tid == 0) shared_mean = 0.0f;
    __syncthreads();
    
    if (tid % warpSize == 0) atomicAdd(&shared_mean, sum);
    __syncthreads();
    
    float mu = shared_mean / cols;
    if (tid == 0) mean_out[row] = mu;

    // 2. Variance
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_x[i] - mu;
        sum_sq += diff * diff;
    }
    sum_sq = warpReduceSum(sum_sq);
    
    if (tid == 0) shared_mean = 0.0f; // Reuse shared
    __syncthreads();
    
    if (tid % warpSize == 0) atomicAdd(&shared_mean, sum_sq);
    __syncthreads();
    
    float var = shared_mean / cols;
    float rstd = rsqrtf(var + eps);
    if (tid == 0) rstd_out[row] = rstd;

    // 3. Normalize and Output
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = (row_x[i] - mu) * rstd;
        
        float g = (gamma) ? gamma[i] : 1.0f;
        float b = (beta) ? beta[i] : 0.0f;
        
        row_y[i] = val * g + b;
    }
}


void layer_norm_forward_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    if (cols > 256) threads = 512;
    if (cols > 512) threads = 1024;
    
    layer_norm_forward_kernel<<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

// =================================================================================
// Backward Kernels
// =================================================================================

// Kernel 1: Compute gradients for Gamma and Beta (Reduce over Rows)
// Grid: [cols], Block: [256]
// Each block handles one column (feature), reduces over all rows.
// Very simple implementation, might be slow for massive batch sizes but fine for GPT-2.
__global__ void ln_backward_gamma_beta_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int rows,
    int cols)
{
    int col = blockIdx.x; // Each block handles one feature dimension
    if (col >= cols) return;
    
    int tid = threadIdx.x;
    
    float d_gamma_acc = 0.0f;
    float d_beta_acc = 0.0f;
    
    for (int row = tid; row < rows; row += blockDim.x) {
        float gy = grad_y[row * cols + col];
        float input_val = x[row * cols + col];
        float m = mean[row];
        float rs = rstd[row];
        
        float norm_x = (input_val - m) * rs;
        
        d_beta_acc += gy;
        d_gamma_acc += gy * norm_x;
    }
    
    // Warp-Block Reduce
    d_gamma_acc = warpReduceSum(d_gamma_acc);
    d_beta_acc = warpReduceSum(d_beta_acc);
    
    __shared__ float s_dgamma, s_dbeta;
    if (tid == 0) { s_dgamma = 0; s_dbeta = 0; }
    __syncthreads();
    
    if (tid % warpSize == 0) {
        atomicAdd(&s_dgamma, d_gamma_acc);
        atomicAdd(&s_dbeta, d_beta_acc);
    }
    __syncthreads();
    
    if (tid == 0) {
        if (grad_gamma) grad_gamma[col] = s_dgamma;
        if (grad_beta) grad_beta[col] = s_dbeta;
    }
}

// Kernel 2: Compute Gradients for Input (Per Row)
// Standard derivation for LayerNorm backward
__global__ void ln_backward_input_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    float* __restrict__ grad_x,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* dy_row = grad_y + row * cols;
    const float* x_row = x + row * cols;
    float* dx_row = grad_x + row * cols;
    
    float m = mean[row];
    float rs = rstd[row];
    
    // 1. Compute local generic reductions: sum(dy * gamma) and sum(dy * gamma * (x-m))
    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_norm = 0.0f;
    
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = (gamma) ? gamma[i] : 1.0f;
        float dy = dy_row[i];
        float val = x_row[i];
        float norm_x = (val - m) * rs;
        
        sum_dy_gamma += dy * g;
        sum_dy_gamma_norm += dy * g * norm_x;
    }
    
    sum_dy_gamma = warpReduceSum(sum_dy_gamma);
    sum_dy_gamma_norm = warpReduceSum(sum_dy_gamma_norm);
    
    __shared__ float s_sum1, s_sum2;
    if (tid == 0) { s_sum1 = 0; s_sum2 = 0; }
    __syncthreads();
    
    if (tid % warpSize == 0) {
        atomicAdd(&s_sum1, sum_dy_gamma);
        atomicAdd(&s_sum2, sum_dy_gamma_norm);
    }
    __syncthreads();
    
    float total_sum1 = s_sum1;
    float total_sum2 = s_sum2;
    
    // 2. Compute dx
    // dxhat = (dy * gamma)
    // dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    //    = rstd * (dy*gamma - (1/D)*sum(dy*gamma) - xhat * (1/D)*sum(dy*gamma*xhat))
    float inv_cols = 1.0f / cols;
    
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = (gamma) ? gamma[i] : 1.0f;
        float dy = dy_row[i];
        float val = x_row[i];
        float norm_x = (val - m) * rs;
        
        float term1 = dy * g;
        float term2 = total_sum1; 
        float term3 = norm_x * total_sum2;
        
        dx_row[i] = rs * (term1 - (term2 + term3) * inv_cols);
    }
}


void layer_norm_backward_cuda(
    const float* grad_y,
    const float* x,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int rows,
    int cols)
{
    // 1. Gradients for Weights (Gamma/Beta)
    if (grad_gamma != nullptr || grad_beta != nullptr) {
        // One block per feature col
        int threads = 256;
        ln_backward_gamma_beta_kernel<<<cols, threads>>>(
            grad_y, x, mean, rstd, grad_gamma, grad_beta, rows, cols
        );
    }
    
    // 2. Gradients for Input
    if (grad_x != nullptr) {
        int threads = 256;
        if (cols > 256) threads = 512;
        ln_backward_input_kernel<<<rows, threads>>>(
            grad_y, x, mean, rstd, gamma, grad_x, cols
        );
    }
}

} // namespace cuda
} // namespace OwnTensor
