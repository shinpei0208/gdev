#include <cuda.h>
#ifdef __KERNEL__ /* just for measurement */
#include <linux/vmalloc.h>
#include <linux/time.h>
#define printf printk
#define malloc vmalloc
#define free vfree
#define gettimeofday(x, y) do_gettimeofday(x)
#else /* just for measurement */
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#endif

#include <gdev_api.h> /* just for GDEV_IPC_RMID */
#define KEY 0x7eadbeef

int copy_to_shm(unsigned int *in, unsigned int size)
{
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	int shmid;

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		return -1;
	}

	res = cuShmGet(&shmid, KEY, size, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuShmGet failed: res = %u\n", res);
		return -1;
	}

	res = cuShmAt(&data_addr, shmid, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuShmAt failed: res = %u\n", res);
		return -1;
	}

	res = cuMemcpyHtoD(data_addr, in, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

	res = cuShmDt(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuShmDt failed: res = %u\n", res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", res);
		return -1;
	}

	return 0;
}

int copy_from_shm(unsigned int *out, unsigned int size)
{
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	int shmid;

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		return -1;
	}

	res = cuShmGet(&shmid, KEY, size, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuShmGet failed: res = %u\n", res);
		return -1;
	}

	res = cuShmAt(&data_addr, shmid, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuShmAt failed: res = %u\n", res);
		return -1;
	}

	res = cuMemcpyDtoH(out, data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}

	res = cuShmDt(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuShmDt failed: res = %u\n", res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", res);
		return -1;
	}

	return 0;
}

int cuda_test_shm(unsigned int size)
{
	int i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	unsigned int *in, *out;
	int shmid;

	in = (unsigned int *) malloc(size);
	out = (unsigned int *) malloc(size);
	for (i = 0; i < size / 4; i++) {
		in[i] = i+1;
		out[i] = 0;
	}
	
	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", res);
		return -1;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		return -1;
	}

	res = cuShmGet(&shmid, KEY, size, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuShmGet failed: res = %u\n", res);
		return -1;
	}

	copy_to_shm(in, size);
	copy_from_shm(out, size);

	res = cuShmCtl(shmid, GDEV_IPC_RMID, NULL);
	if (res != CUDA_SUCCESS) {
		printf("cuShmCtl failed: res = %u\n", res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", res);
		return -1;
	}

	for (i = 0; i < size / 4; i++) {
		if (in[i] != out[i]) {
			printf("in[%d] = %u, out[%d] = %u\n",
				   i, in[i], i, out[i]);
			goto end;
		}
	}

	free(in);
	free(out);

	return 0;

end:
	free(in);
	free(out);

	return -1;
}
