#include <stdint.h>
#include <cuda.h>
__global__
void add(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        int idx = i * n + j;
        c[idx] = a[idx] + b[idx];
    }
}
