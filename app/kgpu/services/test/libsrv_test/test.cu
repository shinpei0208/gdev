#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


__global__ void inc_kernel(int *din, int *dout)
{
    int id = threadIdx.x +  blockIdx.x*blockDim.x;

    dout[id] = din[id]+1;    
}
