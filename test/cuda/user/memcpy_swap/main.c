#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/wait.h>

int main(int argc, char *argv[])
{
	int i;
	pid_t pid;
	int status;
	unsigned int size = 0x10000000; /* 256MB */
	size_t total_mem;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	unsigned int *in, *out;
	
	if (argc > 1)
		sscanf(argv[1], "%x", &size);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}

	res = cuDeviceTotalMem(&total_mem, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceTotalMem failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}
	printf("Total Memory = 0x%x\n", total_mem);

	res = cuMemAlloc(&data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}
	
	pid = fork();

	if (pid == 0) { /* child */
		res = cuInit(0);
		if (res != CUDA_SUCCESS) {
			printf("cuInit failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}
		
		res = cuDeviceGet(&dev, 0);
		if (res != CUDA_SUCCESS) {
			printf("cuDeviceGet failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}
		
		res = cuCtxCreate(&ctx, 0, dev);
		if (res != CUDA_SUCCESS) {
			printf("cuCtxCreate failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}
		
		res = cuMemAlloc(&data_addr, size);
		if (res != CUDA_SUCCESS) {
			printf("cuMemAlloc failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		in = (unsigned int *) malloc(size);
		out = (unsigned int *) malloc(size);
		for (i = 0; i < size / 4; i++) {
			in[i] = i+1;
			out[i] = 0;
		}

		res = cuMemcpyHtoD(data_addr, in, size);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		res = cuMemcpyDtoH(out, data_addr, size);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyDtoH failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		res = cuMemFree(data_addr);
		if (res != CUDA_SUCCESS) {
			printf("cuMemFree failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		res = cuCtxDestroy(ctx);
		if (res != CUDA_SUCCESS) {
			printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		for (i = 0; i < size / 4; i++) {
			if (in[i] != out[i]) {
				printf("in[%d] = %u, out[%d] = %u\n",
					   i, in[i], i, out[i]);
				break;
			}
		}

		free(in);
		free(out);

		printf("Child finished\n");
		exit(-1);
	}
	else { /* parent */
		waitpid(pid, &status, 0);

		in = (unsigned int *) malloc(size);
		out = (unsigned int *) malloc(size);
		for (i = 0; i < size / 4; i++) {
			in[i] = i+1;
			out[i] = 0;
		}

		res = cuMemcpyHtoD(data_addr, in, size);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		res = cuMemcpyDtoH(out, data_addr, size);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyDtoH failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		res = cuMemFree(data_addr);
		if (res != CUDA_SUCCESS) {
			printf("cuMemFree failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		res = cuCtxDestroy(ctx);
		if (res != CUDA_SUCCESS) {
			printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
			exit(-1);
		}

		for (i = 0; i < size / 4; i++) {
			if (in[i] != out[i]) {
				printf("in[%d] = %u, out[%d] = %u\n",
					   i, in[i], i, out[i]);
				break;
			}
		}

		free(in);
		free(out);

		printf("Parent finished\n");
		exit(0);
	}

	return 0;
}
