#ifndef __UTIL_H__
#define __UTIL_H__

#include <cuda.h>
#include <sys/time.h>

CUresult cuda_driver_api_init(CUcontext *pctx, CUmodule *pmod, const char *f);
CUresult cuda_driver_api_exit(CUcontext ctx, CUmodule mod);
void time_measure_start(struct timeval *tv);
void time_measure_end(struct timeval *tv);

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

#if 0
#define CUDA_LAUNCH_KERNEL(f, gdx, gdy, gdz, bdx, bdy
	/* set block sizes. */
	res = cuFuncSetBlockShape(f, bdx, bdy, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape failed: res = %u\n", res);
		return res;
	}

	offset = 0;
	cuParamSetv(f, offset, &d_locations, sizeof(d_locations));
	offset += sizeof(d_locations);
	cuParamSetv(f, offset, &d_distances, sizeof(d_distances));
	offset += sizeof(d_distances);
	cuParamSetv(f, offset, &nr_records, sizeof(nr_records));
	offset += sizeof(nr_records);
	cuParamSetv(f, offset, &lat, sizeof(lat));
	offset += sizeof(lat);
	cuParamSetv(f, offset, &lng, sizeof(lng));
	offset += sizeof(lng);
	cuParamSetSize(f, offset);
	res = cuLaunchGrid(f, gdx, gdy);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchGrid failed: res = %u\n", res);
		return res;
	}
#endif
#endif
