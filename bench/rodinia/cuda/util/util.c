#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#include <cuda.h>
}
#else
#include <cuda.h>
#endif
#include "util.h"

CUresult cuda_driver_api_init(CUcontext *pctx, CUmodule *pmod, const char *f)
{
	CUresult res;
	CUdevice dev;

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuCtxCreate(pctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return res;
	}
	
	res = cuModuleLoad(pmod, f);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		cuCtxDestroy(*pctx);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult cuda_driver_api_exit(CUcontext ctx, CUmodule mod)
{
	CUresult res;

	res = cuModuleUnload(mod);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	return CUDA_SUCCESS;
}

void time_measure_start(struct timeval *tv)
{
	gettimeofday(tv, NULL);
}

void time_measure_end(struct timeval *tv)
{
	struct timeval tv_now, tv_diff;
	double d;

	gettimeofday(&tv_now, NULL);
	tvsub(&tv_now, tv, &tv_diff);

	d = (double) tv_diff.tv_sec * 1000.0 + (double) tv_diff.tv_usec / 1000.0;
	printf("Time (Memory Copy and Launch) = %f (ms)\n", d);
}
