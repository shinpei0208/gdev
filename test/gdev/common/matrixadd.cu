#include <stdint.h>
#include <cuda.h>
extern "C" __global__ void bench(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n&&j<n){
        int idx = i*n+j;
        c[idx] = a[idx] + b[idx];
    }
}
