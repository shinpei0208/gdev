#include "gdev_api.h"
#ifdef __KERNEL__ /* just for measurement */
#define printf printk
#define gettimeofday(x, y) do_gettimeofday(x)
#include <linux/time.h>
#else /* just for measurement */
#include <sys/time.h>
#include <stdio.h>
#endif

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

int gdev_test_memcpy(uint32_t *in, uint32_t *out, uint32_t size, 
					 uint32_t chunk_size, int pipeline_count)
{
	Ghandle handle;
	uint64_t data_addr;
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	unsigned long total;
	struct timeval tv_h2d_start, tv_h2d_end;
	unsigned long h2d;
	struct timeval tv_d2h_start, tv_d2h_end;
	unsigned long d2h;

	gettimeofday(&tv_total_start, NULL);

	if (!(handle = gopen(0))) {
		printf("gopen() failed.\n");
		return -1;
	}

	if (gtune(handle, GDEV_TUNE_MEMCPY_CHUNK_SIZE, chunk_size)) {
		printf("gtune() failed.\n");
		return -1;
	}
	if (gtune(handle, GDEV_TUNE_MEMCPY_PIPELINE_COUNT, pipeline_count)) {
		printf("gtune() failed.\n");
		return -1;
	}

	if (!(data_addr = gmalloc(handle, size))) {
		printf("gmalloc() failed.\n");
		return -1;
	}

	gettimeofday(&tv_h2d_start, NULL);
	gmemcpy_to_device(handle, data_addr, in, size);
	gettimeofday(&tv_h2d_end, NULL);


	gettimeofday(&tv_d2h_start, NULL);
	gmemcpy_from_device(handle, out, data_addr, size);
	gettimeofday(&tv_d2h_end, NULL);

	gfree(handle, data_addr);
	gclose(handle);

	gettimeofday(&tv_total_end, NULL);

	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	printf("%lu, ", h2d);
	printf("%lu", d2h);
	printf("\n");

	return 0;
}
