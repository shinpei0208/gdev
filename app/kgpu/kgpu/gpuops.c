/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
#include "gputils.h"

void gpu_init();
void gpu_finit();

void *gpu_alloc_pinned_mem(unsigned long size);
void gpu_free_pinned_mem(void *p);

void gpu_pin_mem(void *p, size_t sz);
void gpu_unpin_mem(void *p);

int gpu_alloc_device_mem(struct kgpu_service_request *sreq);
void gpu_free_device_mem(struct kgpu_service_request *sreq);

int gpu_execution_finished(struct kgpu_service_request *sreq);
int gpu_post_finished(struct kgpu_service_request *sreq);

struct kgpu_gpu_mem_info devbuf;
struct kgpu_gpu_mem_info devbuf4vma;

void gpu_init()
{
	CUresult res;

	res = cuMemAlloc((CUdeviceptr*)&devbuf.uva, KGPU_BUF_SIZE);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (devbuf) failed: res = %u\n", res);
		return ;
	}

	res = cuMemAlloc((CUdeviceptr*)&devbuf4vma.uva, KGPU_BUF_SIZE);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (devbuf4vma failed: res = %u\n", res);
		return ;
	}
}

void gpu_finit()
{
	cuMemFree((CUdeviceptr)devbuf.uva);
	cuMemFree((CUdeviceptr)devbuf4vma.uva);
}

void *gpu_alloc_pinned_mem(unsigned long size)
{
	void *h;
	cuMemAllocHost(&h, size);
	return h;
}

void gpu_free_pinned_mem(void* p)
{
	cuMemFreeHost(p);
}

void gpu_pin_mem(void *p, size_t sz)
{
	size_t rsz = round_up(sz, PAGE_SIZE);
	printf("cuMemHostRegister not supported\n");
	//cuMemHostRegister(p, rsz, CU_MEMHOSTREGISTER_PORTABLE);
}

void gpu_unpin_mem(void *p)
{
	printf("cuMemHostUnregister not supported\n");
	//cuMemHostUnregister(p);
}

int gpu_execution_finished(struct kgpu_service_request *sreq)
{
	return 1;
}

int gpu_post_finished(struct kgpu_service_request *sreq)
{
	return 1;
}

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static int __merge_2ranges(
	unsigned long r1, unsigned long s1, unsigned long r2, unsigned long s2,
	unsigned long *e, unsigned long *s) 
{
	// r1   r2
	if (r1 < r2) {
		if (r1+s1 >= r2) {
			*e = r1;
			*s = max(r1+s1, r2+s2) - r1;
			return 1;
		}
		
		return 0;
	} else if (r1 == r2) {
		*e = r1;
		*s = max(s1, s2);
		return 1;
	} else {
		// r2  r1
		if (r2+s2 >= r1) {
			*e = r2;
			*s = max(r1+s1, r2+s2) - r2;
			return 1;
		}
		
		return 0;
	}
}


static int __merge_ranges(unsigned long ad[], unsigned long sz[], int n)
{
	int i;
	
	for (i=0; i<n; i++) {
		ad[i] = round_down(ad[i], PAGE_SIZE);
		sz[i] = round_up(sz[i], PAGE_SIZE);
	}
	
	switch(n) {
		case 0:
			return 0;
		case 1:
			return 1;
		case 2:
			if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0]))
				return 1;
			else
				return 2;
		case 3:
			if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0])) {
				if (__merge_2ranges(ad[0], sz[0], ad[2], sz[2], &ad[0], &sz[0])) {
					return 1;
				} else {
					ad[1] = ad[2];
					sz[1] = sz[2];
					return 2;
				}
			} else if (__merge_2ranges(ad[0], sz[0], ad[2], sz[2], &ad[0], &sz[0])) {
				if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0])) {
					return 1;
				} else
					return 2;
			} else if (__merge_2ranges(ad[2], sz[2], ad[1], sz[1], &ad[1], &sz[1])) {
				if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0]))
					return 1;
				else
					return 2;
			} else
				return 3;
		   
		default:
			return 0;
	}
	
	// should never reach here
	//return 0;
}

/*
 * A little bit old policy, but may give you a brief pic of what
 * K-U mm does.
 *
 * Allocation policy is simple here: copy what the kernel part does
 * for the GPU memory. This works because:
 *   - GPU memory and host memory are identical in size
 *   - Whenever a host memory region is allocated, the same-sized
 *	 GPU memory must be used for its GPU computation.
 *   - The data field in ku_request also uses pinned memory but we
 *	 won't allocate GPU memory for it cause it is just for
 *	 service provider. This is fine since the data tend to be
 *	 very tiny.
 */
int gpu_alloc_device_mem(struct kgpu_service_request *sreq)
{
	unsigned long pin_addr[3] = {0,0,0}, pin_sz[3] = {0,0,0};
	int npins = 0, i;
	
	if (ADDR_WITHIN(sreq->hin, hostbuf.uva, hostbuf.size))
		sreq->din =
			(void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hin);
	else {
		sreq->din =
			(void*)ADDR_REBASE(devbuf4vma.uva, hostvma.uva, sreq->hin);
		
		pin_addr[npins] = TO_UL(sreq->hin);
		pin_sz[npins] = sreq->insize;
		npins++;
	}
	
	if (ADDR_WITHIN(sreq->hout, hostbuf.uva, hostbuf.size))
		sreq->dout =
			(void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hout);
	else {
		sreq->dout =
			(void*)ADDR_REBASE(devbuf4vma.uva, hostvma.uva, sreq->hout);
		
		pin_addr[npins] = TO_UL(sreq->hout);
		pin_sz[npins] = sreq->outsize;
		npins++;
	}
	
	if (ADDR_WITHIN(sreq->hdata, hostbuf.uva, hostbuf.size))
		sreq->ddata =
			(void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hdata);
	else if (ADDR_WITHIN(sreq->hdata, hostvma.uva, hostvma.size)){
		sreq->ddata =
			(void*)ADDR_REBASE(devbuf4vma.uva, hostvma.uva, sreq->hdata);
		
		pin_addr[npins] = TO_UL(sreq->hdata);
		pin_sz[npins] = sreq->datasize;
		npins++;
	}
	
	npins = __merge_ranges(pin_addr, pin_sz, npins);
	for (i=0; i<npins; i++) {
		gpu_pin_mem((void*)pin_addr[i], pin_sz[i]);
	}
	
	return 0;
}

void gpu_free_device_mem(struct kgpu_service_request *sreq)
{
	unsigned long pin_addr[3] = {0,0,0}, pin_sz[3] = {0,0,0};
	int npins = 0, i;
	
	sreq->din = NULL;
	sreq->dout = NULL;
	sreq->ddata = NULL;   
	
	if (ADDR_WITHIN(sreq->hin, hostvma.uva, hostvma.size)) {
		pin_addr[npins] = TO_UL(sreq->hin);
		pin_sz[npins] = sreq->insize;
		npins++;
	}
	if (ADDR_WITHIN(sreq->hout, hostvma.uva, hostvma.size)) {
		pin_addr[npins] = TO_UL(sreq->hout);
		pin_sz[npins] = sreq->outsize;
		npins++;
	}
	if (ADDR_WITHIN(sreq->hdata, hostvma.uva, hostvma.size)) {
		pin_addr[npins] = TO_UL(sreq->hdata);
		pin_sz[npins] = sreq->datasize;
		npins++;
	}
	
	npins = __merge_ranges(pin_addr, pin_sz, npins);
	for (i=0; i<npins; i++) {
		gpu_unpin_mem((void*)pin_addr[i]);
	}
}

int default_compute_size(struct kgpu_service_request *sreq)
{
	sreq->block_x = 32;
	sreq->block_y = 1;
	sreq->grid_x = 512;
	sreq->grid_y = 1;
	return 0;
}

int default_prepare(struct kgpu_service_request *sreq)
{
	cuMemcpyHtoD( (CUdeviceptr)sreq->din, sreq->hin, sreq->insize);
	return 0;
}

int default_post(struct kgpu_service_request *sreq)
{
	cuMemcpyDtoH( sreq->hout, (CUdeviceptr)sreq->dout, sreq->outsize);
	return 0;
}
