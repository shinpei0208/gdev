#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/wait.h>

/* tvsub: ret = x - y. */
static inline void tvsub(struct timeval *x, 
						 struct timeval *y, 
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

void test_tasks(unsigned int size, int nr_tasks)
{
	int i;
	pid_t pid;
	int status;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	CUmodule module;
	CUfunction function;
	unsigned int *in, *out;
	unsigned int n = size / 4;

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", res);
		exit(-1);
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		exit(-1);
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		exit(-1);
	}

	res = cuMemAlloc(&data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		exit(-1);
	}

	in = (unsigned int *) malloc(size);
	out = (unsigned int *) malloc(size);
	
	res = cuMemcpyHtoD(data_addr, in, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		exit(-1);
	}

#if 1
	res = cuModuleLoad(&module, "./loop_gpu.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		exit(-1);
	}
	res = cuModuleGetFunction(&function, module, "_Z4loopPjjj");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		exit(-1);
	}
	
	void *param1[] = {&data_addr, &size, &n}; 
	//res = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, (void**)param1, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel failed: res = %u\n", res);
		exit(-1);
	}
	//cuCtxSynchronize();
#endif

	if (--nr_tasks) {
		pid = fork();
		if (pid == 0) { /* child */
			test_tasks(size, nr_tasks);
			printf("Child finished\n");
			exit(0);
		}
		else { /* parent */
			waitpid(pid, &status, 0);
		}
	}

#if 0
	res = cuModuleLoad(&module, "./loop_gpu.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		exit(-1);
	}
	res = cuModuleGetFunction(&function, module, "_Z4loopPjjj");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		exit(-1);
	}
	
	void *param1[] = {&data_addr, &size, &n}; 
	res = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, (void**)param1, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel failed: res = %u\n", res);
		exit(-1);
	}
#endif

	res = cuMemcpyDtoH(out, data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", res);
		exit(-1);
	}

	res = cuModuleUnload(module);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleUnload failed: res = %lu\n", res);
		exit(-1);
	}
	
	res = cuMemFree(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %u\n", res);
		exit(-1);
	}
	
	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}
	
	free(in);
	free(out);
}

int main(int argc, char *argv[])
{
	int i;
	pid_t pid;
	int status;
	unsigned int size = 0x10000000; /* 256MB */
	int nr_tasks = 2;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	struct timeval tv_start, tv_end, tv;
	float makespan;
	
	if (argc > 1)
		sscanf(argv[1], "%x", &size);
	if (argc > 2)
		sscanf(argv[2], "%d", &nr_tasks);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", res);
		exit(-1);
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		exit(-1);
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		exit(-1);
	}

#if 0
	/* alloc big memory at the beginning. */
	res = cuMemAlloc(&data_addr, 0x20000000); /* 512MB */
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		exit(-1);
	}
#endif	
	gettimeofday(&tv_start, NULL);
	pid = fork();
	if (pid == 0) { /* child */
		test_tasks(size, nr_tasks);
		printf("Child finished\n");
		exit(0);
	}
	else { /* parent */
		waitpid(pid, &status, 0);
	}
	gettimeofday(&tv_end, NULL);

	tvsub(&tv_end, &tv_start, &tv);
	makespan = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Makespan: %f\n", makespan);

	res = cuMemFree(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}

#if 0	
	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
		exit(-1);
	}
#endif
	printf("Root parent finished\n");

	return 0;
}
