/*
 * Copyright (C) 2013 Sylvain Collange <sylvain.collange@inria.fr>
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <assert.h>
#include <stdio.h>
#include "gdev_api.h"
#include "gdev_device.h"
#include "gdev_io_memcpy.h"
#include "gdev_nvidia.h"
#include "barra_gdev.h"

/**
 * OS driver and user-space runtime depen functions.
 */

#define GDEV_DEVICE_MAX_COUNT 32

struct gdev_device *lgdev; /* local gdev_device structure for user-space scheduling */

int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *result)
{
	switch (type) {
	case GDEV_NVIDIA_QUERY_MP_COUNT:
		*result = barra_get_attribute(/* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */ 16, gdev->id);
		break;
	case GDEV_QUERY_DEVICE_MEM_SIZE:
		*result = 0x40000000;	/* Hardcoded to 1 GB */
		break;
	case GDEV_QUERY_DMA_MEM_SIZE:
		/* XXX */
		goto fail;
	case GDEV_QUERY_CHIPSET:
		*result = 0x50 | 0x8000;		/* NV50 (G80) | barra_flag(0x8000 */
		break;
	case GDEV_QUERY_PCI_VENDOR:
		*result = 0x10de;	/* NVIDIA */
		break;
	case GDEV_QUERY_PCI_DEVICE:
		*result = 0x05e1;	/* GT200 */
		break;
	default:
		goto fail;
	}

	return 0;

fail:
	GDEV_PRINT("Failed to query %u\n", type);
	return -EINVAL;
}

struct gdev_device *gdev_raw_dev_open(int minor)
{
	struct gdev_device *gdev;
	int major, max;
	void * priv = 0;

	if (!gdevs) {
#ifndef GDEV_SCHED_DISABLED
		gdevs = (struct gdev_device *)gdev_attach_shms_dev(GDEV_DEVICE_MAX_COUNT); /* FIXME: constant number   */
		if (!gdevs)
			return NULL;
		minor++;
#else
	    	gdevs = malloc(sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT);
		if (!gdevs)
			return NULL;
		memset(gdevs, 0, sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT);
#endif
	}

	gdev = &gdevs[minor];
	major = 0;
	max = 0;
	while( minor > max + VCOUNT_LIST[major] )
	    max += VCOUNT_LIST[major++];

	/* init Barra */
	barra_dev_open();

	if (gdev->users == 0) {
#ifdef GDEV_SCHED_DISABLED
		gdev_init_device(gdev, minor, (void *)priv);
#else
		lgdev = MALLOC(sizeof(*lgdev));
		memset(lgdev, 0, sizeof(*lgdev));
		gdev_init_device(lgdev, major, (void *)priv);
		gdev_init_device(gdevs, major, (void *)priv);
		gdev_init_virtual_device(gdev, minor, 100, (void *)ADDR_SUB(gdev,gdevs));
	}else{
		lgdev = MALLOC(sizeof(*lgdev));
		memset(lgdev, 0, sizeof(*lgdev));
		gdev_init_device(lgdev, major, (void *)priv);
#endif
	}
	gdev->users++;

	return gdev;
}

/* close the specified Gdev object. */
void gdev_raw_dev_close(struct gdev_device *gdev)
{
	int i;

	gdev->users--;

	if (gdev->users == 0) {
		gdev_exit_device(gdev);
		/* TODO: cleanup Barra dev */
		barra_dev_close();
		for (i = 0; i < GDEV_DEVICE_MAX_COUNT; i++) {
			if (gdevs[i].users > 0)
				return;
		}
		FREE(gdevs);
		gdevs = NULL;
	}
}

/* allocate a new virual address space object.  */
struct gdev_vas *gdev_raw_vas_new(struct gdev_device *gdev, uint64_t size)
{
	struct gdev_vas *vas;



#ifndef GDEV_SCHED_DISABLED
	if (!(vas = gdev_attach_shms_vas(0)))
#else
	if (!(vas = MALLOC(sizeof(*vas))))
#endif
		goto fail_vas;

	/* TODO: creat Barra VAS */

	/* private data */
	/*vas->pvas = nvas;*/

	return vas;

  /* fail_nvas: */
	FREE(vas);
fail_vas:
	return NULL;
}

/* free the specified virtual address space object. */
void gdev_raw_vas_free(struct gdev_vas *vas)
{
	/* TODO free Barra VAS */
	FREE(vas);
}

/* create a new GPU context object. */
struct gdev_ctx *gdev_raw_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	/* struct barra_ib_bo *fence_bo; */

	printf("Trying to get ctx in barra\n");
	if (!(ctx = malloc(sizeof(*ctx))))
		goto fail_ctx;
	memset(ctx, 0, sizeof(*ctx));

	CUcontext bctx = barra_ctx_new(gdev->id);
	if (bctx == NULL)
		goto fail_ctx;
	ctx->pctx = bctx;

	/* Skip FIFO setup */


	/* fence buffer. */
	ctx->fence.map = malloc(GDEV_FENCE_BUF_SIZE);
	ctx->fence.seq = 0;




	return ctx;
fail_ctx:
	return NULL;
}

/* destroy the specified GPU context object. */
void gdev_raw_ctx_free(struct gdev_ctx *ctx)
{
	barra_ctx_free(ctx->pctx);
	FREE(ctx);
}

/* allocate a new device memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t size)
{
	struct gdev_mem *mem;
	int64_t addr;

#ifndef GDEV_SCHED_DISABLED
	if (!(mem = (struct gdev_mem *)gdev_attach_shms_mem(0)))
#else
	if (!(mem = (struct gdev_mem *) MALLOC(sizeof(*mem))))
#endif
	    	goto fail_mem;

	/* address, size, and map. */
	addr = barra_mem_alloc(size);
	if(!addr)
		goto fail_alloc;

	mem->addr = addr;
	mem->size = size;
	mem->map = NULL;

	return mem;

fail_alloc:
	GDEV_PRINT("Failed to allocate device memory.\n");
	free(mem);
fail_mem:
	return NULL;
}

/* allocate a new host DMA memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc_dma(struct gdev_vas *vas, uint64_t size)
{
	/* Host DMA memory is emulated by regular host memory */
	struct gdev_mem *mem;
	void * addr;

#ifndef GDEV_SCHED_DISABLED
	if (!(mem = (struct gdev_mem *)gdev_attach_shms_mem(0)))
#else
	if (!(mem = (struct gdev_mem *) MALLOC(sizeof(*mem))))
#endif
	    	goto fail_mem;

	/* address, size, and map. */
	addr = malloc(size);
	if(!addr)
		goto fail_alloc;

	/* Hack: we map "host DMA" memory in device space by setting msb of the host address */
	assert(!((uintptr_t)addr & 0x8000000000000000ull));
	mem->addr = (uintptr_t)addr | 0x8000000000000000ull;
	mem->size = size;
	mem->map = addr;

	return mem;

fail_alloc:
	GDEV_PRINT("Failed to allocate host memory.\n");
	free(mem);
fail_mem:
	return NULL;
}

/* free the specified memory object. */
void gdev_raw_mem_free(struct gdev_mem *mem)
{
	if(mem->map) {
		/* This is host memory */
		free((void*)mem->map);
	}
	else {
		/* This is device memory */
		barra_mem_free(mem->addr);
	}
	FREE(mem);
}

/* allocate a reserved swap memory object. size may be aligned. */
struct gdev_mem *gdev_raw_swap_alloc(struct gdev_device *gdev, uint64_t size)
{
	GDEV_PRINT("Swap memory not implemented\n");
	/* To be implemented. */
	return NULL;
}

/* free the specified swap memory object. */
void gdev_raw_swap_free(struct gdev_mem *mem)
{
	GDEV_PRINT("Swap memory not implemented\n");
	/* To be implemented. */
}

/* create a new memory object sharing memory space with @mem. */
struct gdev_mem *gdev_raw_mem_share(struct gdev_vas *vas, struct gdev_mem *mem)
{
	GDEV_PRINT("Shared memory not implemented\n");
	/* To be implemented. */
	return NULL;
}

/* destroy the memory object by just unsharing memory space. */
void gdev_raw_mem_unshare(struct gdev_mem *mem)
{
	GDEV_PRINT("Shared memory not implemented\n");
	/* To be implemented. */
}

/* map device memory to host DMA memory. */
void *gdev_raw_mem_map(struct gdev_mem *mem)
{
	GDEV_PRINT("Memory mapping not implemented\n");
	return NULL;
}

/* unmap device memory from host DMA memory. */
void gdev_raw_mem_unmap(struct gdev_mem *mem, void *map)
{
	GDEV_PRINT("Memory mapping not implemented\n");
}

/* get physical bus address. */
uint64_t gdev_raw_mem_phys_getaddr(struct gdev_mem *mem, uint64_t offset)
{
	GDEV_PRINT("Memory mapping not implemented\n");
	return 0;
}

uint32_t gdev_raw_read32(struct gdev_mem *mem, uint64_t addr)
{
	GDEV_PRINT("%s is not implemented\n", __func__);
	return 0;
}

void gdev_raw_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val)
{
	GDEV_PRINT("%s is not implemented\n", __func__);
}

int gdev_raw_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size)
{
	GDEV_PRINT("%s is not implemented\n", __func__);
	return -ENOSYS;
}

int gdev_raw_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size)
{
	GDEV_PRINT("%s is not implemented\n", __func__);
  return -ENOSYS;
}
