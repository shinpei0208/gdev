#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define N 3

__global__ void multiply( unsigned int *a, unsigned int *b, unsigned int *c )
{
	int id = blockIdx.x * N + threadIdx.x; // blockDim.x doesn't work...
	int row = blockIdx.x;
	int col = threadIdx.x;
	unsigned int tmp  = 0;

	for(int i = 0; i < N; i++){
		tmp += a[row*N+i] * b[col+i*N];
	}
	c[id] = tmp;
}

unsigned int a[] = {1,2,3,4,5,6,7,8,9};
unsigned int b[] = {2,0,0, 0,2,0, 0,0,2};
unsigned int c[] = {0xff,0xff,0xff, 0xff,0xff,0xff, 0xff,0xff,0xff};

int test_matrixmul(void)
{
	int *devA, *devB, *devC;
	cudaMalloc((void**)&devA, sizeof(a));
	cudaMalloc((void**)&devB, sizeof(b));
	cudaMalloc((void**)&devC, sizeof(c));

	cudaMemcpy(devA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, sizeof(b), cudaMemcpyHostToDevice);
	cudaMemcpy(devC, c, sizeof(c), cudaMemcpyHostToDevice);

	dim3 grid; grid.x = 3; grid.y = 1; grid.z = 1;
	dim3 block; block.x = 3; block.y = 1; block.z = 1;
	cudaConfigureCall(grid, block, 0, 0);

	cudaSetupArgument(&devA, 8, 0);
	cudaSetupArgument(&devB, 8, 8);
	cudaSetupArgument(&devC, 8, 16);

	cudaLaunch("_Z8multiplyPjS_S_");
	memset(c, 0, 9*4);

	cudaMemcpy(a, devA, sizeof(a), cudaMemcpyDeviceToHost);
	printf("A: %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
		   a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
	cudaMemcpy(b, devB, sizeof(b), cudaMemcpyDeviceToHost);
	printf("B: %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
		   b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]);
	cudaMemcpy(c, devC, sizeof(c), cudaMemcpyDeviceToHost);
	printf("C: %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
		   c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return 0;
}

int test_load_store(void)
{
	return 0;
}


int main(void)
{
	test_matrixmul();
	printf("test finished\n");
	return 0;
}
