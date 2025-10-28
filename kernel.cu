#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

extern "C" void add_launch(const float* a, const float* b, float* c, int n){
    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    add_kernel<<<gs, bs>>>(a, b, c, n);
    cudaDeviceSynchronize();
}