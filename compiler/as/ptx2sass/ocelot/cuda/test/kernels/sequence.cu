/*!
	\brief sequence.cu
	\author Andrew Kerr

	\brief simple test of a CUDA implementation's ability to allocate memory on the device, launch
		a kernel, and fetch its results. One kernel requires no syncthreads, another kernel requires
		one synchronization
*/

#include <stdio.h>
#include <dlfcn.h>

#if 1

extern "C" __global__ void simple(int *A) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	A[i] = i;
}

extern "C" __global__ void sequence(int *A, int N) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N) {
		A[i] = 2*i;
	}
}

extern "C" __global__ void testShareSimple(int *A) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ int Share[32];
	
	int a = A[i];
	Share[threadIdx.x] = a;
	__syncthreads();
	A[i] = Share[threadIdx.x ^ 1];
	A[i] = Share[31 - threadIdx.x];
}

extern "C" __global__ void v4sequence(int4 *A, int N) {
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int4 b = make_int4(i, 2*i, 3*i, 4*i);
	A[i-1] = b;
}

#endif

extern "C" __global__ void testShr(int *A) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int b;
	__shared__ int storage[256];
	
	storage[threadIdx.x] = 2*i;
	__syncthreads();
	if (i & 1) {
		b = storage[threadIdx.x ^ 1] * 19;
	}
	else {
		b = storage[threadIdx.x ^ 1] * 13;
	}
	A[i] = b;
}

int main(int argc, char *arg[]) {

	const int BlockSize = 4;
	const int N = 4;
	int *A_host, *A_gpu =0;
	int errors = 0;

	size_t bytes = sizeof(int)*N;
	
	cudaError_t result = cudaThreadSynchronize();
	if (result != cudaSuccess) {
		printf("cudaThreadSynchronize() = %s\n", cudaGetErrorString(result));
		printf("Failed to load CUDA library:\n%s\n", dlerror());
		return 0;
	}

	if (cudaMalloc((void **)&A_gpu, bytes) != cudaSuccess) {
		printf("cudaMalloc() - failed to allocate %d bytes on device\n", (int)bytes);
		return -1;
	}

	A_host = (int *)malloc(bytes);
	for (int i = 0; i < N; i++) {
		A_host[i] = -1;
	}
	
	cudaMemcpy(A_gpu, A_host, bytes, cudaMemcpyHostToDevice);
	
	dim3 grid((N+BlockSize-1)/BlockSize,1);
	dim3 block(BlockSize, 1);
	
	simple<<< grid, block >>>(A_gpu);
	
	sequence<<< grid, block >>>(A_gpu, N);
	
	cudaMemcpy(A_host, A_gpu, bytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N && errors < 5; i++) {
		if (A_host[i] != 2*i) {
			
			printf("ERROR 1 [%d] - expected: %d, got: %d\n", i, 2*i, A_host[i]);
			++errors;
		}
	}
#if 0
	grid.x /= 4;
	v4sequence<<< grid, block >>>((int4 *)A_gpu, N/4);
	cudaMemcpy(A_host, A_gpu, bytes, cudaMemcpyDeviceToHost);
	grid.x *= 4;


	if (!errors) {
		sequence<<< grid, block >>>(A_gpu, N);
		
		printf("\n\n\n\n\n\n");
		
		testShareSimple<<< grid, block >>>(A_gpu);
		if (cudaMemcpy(A_host, A_gpu, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
			printf("cudaMemcpy(A, A) - failed to copy %d bytes from device to host\n", (int)bytes);
			cudaFree(A_gpu);
			free(A_host);
		}
	
		for (int i = 0; i < N; i++) {
//			int p = i + 31 - 2 * (i % 32);
			int p = i;
			if (p & 0x01) {
				p --;
			}
			else {
				p ++;
			}
			int expected = p * 2;
			int got = A_host[i];
			if (expected != got) {
				printf("ERROR 2 [%d] - expected: %d, got: %d\n", i, expected, got);
				++errors;
			}
		}
	}
#endif
	
	if (!errors) {

		testShr<<< grid, block >>>(A_gpu);
	
		if (cudaMemcpy(A_host, A_gpu, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
			printf("cudaMemcpy(A, B) - failed to copy %d bytes from device to host\n", (int)bytes);
			cudaFree(A_gpu);
			free(A_host);
		}
	
		for (int i = 0; (errors < 5) && i < N; ++i) {
			int b;
			if (i & 1) {
				b = (i ^ 1) * 2 * 19;
			}
			else {
				b = (i ^ 1) * 2 * 13;
			}
			int got = A_host[i];
			if (b != got) {
				printf("ERROR 3 [%d] - expected: %d, got: %d\n", i, b, got);
				++errors;
			}
		}
	}

	cudaFree(A_gpu);
	free(A_host);


	if (errors) {
		printf("Pass/Fail : Fail\n");
	}
	else {
		printf("Pass/Fail : Pass\n");
	}

	return 0;
}

