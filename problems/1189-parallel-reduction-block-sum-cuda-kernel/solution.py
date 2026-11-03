#include <cuda_runtime.h>

__global__ void reduce_kernel(
    const float* input,
    float* partial,
    int N
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    if (idx < N)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    // Store block sum
    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}


__global__ void final_reduce(
    const float* partial,
    float* output,
    int num_blocks
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    if (tid < num_blocks)
        sdata[tid] = partial[tid];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[0] = sdata[0];
}


void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        *output = 0.0f;
        return;
    }

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float *d_input, *d_partial, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input to device
    cudaMemcpy(
        d_input, input,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // First-stage reduction
    reduce_kernel<<<blocks, threads>>>(
        d_input, d_partial, N
    );

    // Second-stage reduction
    final_reduce<<<1, threads>>>(
        d_partial, d_output, blocks
    );

    // Copy final result to host
    cudaMemcpy(
        output, d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaFree(d_output);
}
