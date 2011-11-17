/*
 * Copyright 2011 Shinpei Kato
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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gdev_list.h"
#include "gdev_nvidia.h"
#include "gdev_time.h"

void gdev_heap_init(gdev_vas_t *vas)
{
	__gdev_list_init(&vas->mem_list, NULL); /* device memory list. */
	__gdev_list_init(&vas->dma_mem_list, NULL); /* host dma memory list. */
}

/* add the device memory object to the memory list. */
void gdev_heap_add(gdev_mem_t *mem, int type)
{
	gdev_vas_t *vas = mem->vas;

	switch (type) {
	case GDEV_MEM_DEVICE:
		__gdev_list_add(&mem->list_entry, &vas->mem_list);
		break;
	case GDEV_MEM_DMA:
		__gdev_list_add(&mem->list_entry, &vas->dma_mem_list);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
}

/* delete the device memory object from the memory list. */
void gdev_heap_del(gdev_mem_t *mem)
{
	__gdev_list_del(&mem->list_entry);
}

/* look up the memory object allocated at the specified address. */
gdev_mem_t *gdev_heap_lookup(gdev_vas_t *vas, uint64_t addr, int type)
{
	gdev_mem_t *mem;
	gdev_list_t *entry;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_list_for_each (mem, entry, &vas->mem_list) {
			if (mem && mem->addr == addr)
				return mem;
		}
		break;
	case GDEV_MEM_DMA:
		gdev_list_for_each (mem, entry, &vas->dma_mem_list) {
			if (mem && (uint64_t)mem->map == addr)
				return mem;
		}
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}

	return NULL;
}

/* copy data of @size from @src_addr to @dst_addr. */
uint32_t gdev_memcpy
(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
	gdev_vas_t *vas = ctx->vas;
	gdev_device_t *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t sequence = ++ctx->fence.sequence[GDEV_FENCE_DMA];

	/* it's important to emit a fence *before* memcpy():
	   the EXEC method of the PCOPY and M2MF engines is associated with
	   the QUERY method, i.e., if QUERY is set, the sequence will be 
	   written to the specified address when the data are transfered. */
	compute->fence_write(ctx, GDEV_FENCE_DMA, sequence);
	compute->memcpy(ctx, dst_addr, src_addr, size);

	return sequence;
}

/* launch the kernel onto the GPU. */
uint32_t gdev_launch(gdev_ctx_t *ctx, struct gdev_kernel *kern)
{
	gdev_vas_t *vas = ctx->vas;
	gdev_device_t *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t seq = ++ctx->fence.sequence[GDEV_FENCE_COMPUTE];

	/* it's important to emit a fence *after* launch():
	   the LAUNCH method of the PGRAPH engine is not associated with
	   the QUERY method, i.e., we have to submit the QUERY method 
	   explicitly after the kernel is launched. */
	compute->launch(ctx, kern);
	compute->fence_write(ctx, GDEV_FENCE_COMPUTE, seq);
	
	return seq;
}

/* barrier memory access. */
void gdev_mb(gdev_ctx_t *ctx)
{
	gdev_vas_t *vas = ctx->vas;
	gdev_device_t *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;

	compute->membar(ctx);
}

/* poll until the resource becomes available. */
int gdev_poll(gdev_ctx_t *ctx, int type, uint32_t seq, gdev_time_t *timeout)
{
	gdev_time_t time_start, time_now, time_elapse, time_relax;
	gdev_vas_t *vas = ctx->vas;
	gdev_device_t *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t val;

	gdev_time_stamp(&time_start);
	gdev_time_sec(&time_relax, 1); /* relax polling when 1 second elapsed. */

	compute->fence_read(ctx, type, &val);

	while (val < seq || val > seq + GDEV_FENCE_LIMIT) {
		gdev_time_stamp(&time_now);
		gdev_time_sub(&time_elapse, &time_now, &time_start);
		/* relax polling after some time. */
		if (gdev_time_ge(&time_elapse, &time_relax)) {
			SCHED_YIELD();
		}
		compute->fence_read(ctx, type, &val);
		/* check timeout. */
		if (timeout && gdev_time_ge(&time_elapse, timeout))
			return -ETIME;
	}

	/* sequence rolls back to zero, if necessary. */
	if (ctx->fence.sequence[type] == GDEV_FENCE_LIMIT) {
		ctx->fence.sequence[type] = 0;
	}

	return 0;
}
