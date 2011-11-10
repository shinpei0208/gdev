#define PTX

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>

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

int gdev_test_memcpy(uint32_t *in, uint32_t *out, uint32_t size)
{
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	unsigned long total;
	struct timeval tv_h2d_start, tv_h2d_end;
	unsigned long h2d;
	struct timeval tv_d2h_start, tv_d2h_end;
	unsigned long d2h;

	gettimeofday(&tv_total_start, NULL);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", res);
		return 0;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		return 0;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		return 0;
	}

	cuMemAlloc(&data_addr, size);

	gettimeofday(&tv_h2d_start, NULL);
	res = cuMemcpyHtoD(data_addr, in, size);
	gettimeofday(&tv_h2d_end, NULL);

	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return 0;
	}

	gettimeofday(&tv_d2h_start, NULL);
	res = cuMemcpyDtoH(out, data_addr, size);
	gettimeofday(&tv_d2h_end, NULL);

	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyDtoH failed: res = %u\n", res);

	cuMemFree(data_addr);

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", res);
		return 0;
	}

	gettimeofday(&tv_total_end, NULL);

	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	//printf("HtoD: %lu\n", h2d);
	//printf("DtoH: %lu\n", d2h);
	//printf("%lu\n", total - h2d - d2h);
	//printf("Total = %lu ms\n", total);

	return 0;
}

int main(int argc, char *argv[])
{
	uint32_t *in, *out;
	uint32_t size = 0x10000000;
	uint32_t ch_size = 0x2000000;
	int pl = 1;
	int i, tmp;

	for (i = 1; i < argc; i++) {
		if (strncmp(argv[i], "--chunk", (tmp = strlen("--chunk"))) == 0) {
			if (argv[i][tmp] != '=') {
				printf("option \"%s\" is invalid.\n", argv[i]);
				exit(1);
			}
			sscanf(&argv[i][tmp+1], "%x", &ch_size);
		}
		else if (strncmp(argv[i], "--data", (tmp = strlen("--data"))) == 0) {
			if (argv[i][tmp] != '=') {
				printf("option \"%s\" is invalid.\n", argv[i]);
				exit(1);
			}
			sscanf(&argv[i][tmp+1], "%x", &size);
		}
		else if (strncmp(argv[i], "--pl", (tmp = strlen("--pl"))) == 0) {
			if (argv[i][tmp] != '=') {
				printf("option \"%s\" is invalid.\n", argv[i]);
				exit(1);
			}
			sscanf(&argv[i][tmp+1], "%d", &pl);
		}
	}

	printf("size = 0x%x\n", size);
	printf("ch_size = 0x%x\n", ch_size);
	in = (uint32_t *) malloc(size);
	out = (uint32_t *) malloc(size);
	for (i = 0; i < size / 4; i++) {
		in[i] = i+1;
		out[i] = 0;
	}
	
	gdev_test_memcpy(in, out, size);
	
	for (i = 0; i < size / 4; i++) {
		if (in[i] != out[i]) {
			printf("in[%d] = %lu, out[%d] = %lu\n",
				   i, in[i], i, out[i]);
			printf("Test failed.\n");
			goto end;
		}
	}
	free(in);
	free(out);

	return 0;

end:
	free(in);
	free(out);
	
	return 0;
}
