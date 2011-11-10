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

#include "gdev_nvidia.h"

/* add the device memory object to the memory list. */
void gdev_heap_add(gdev_mem_t *mem)
{
	gdev_vas_t *vas = mem->vas;

	__gdev_list_add(&mem->list_entry, &vas->memlist);
}

/* delete the device memory object from the memory list. */
void gdev_heap_del(gdev_mem_t *mem)
{
	__gdev_list_del(&mem->list_entry);
}

/* look up the memory object allocated at the specified address. */
gdev_mem_t *gdev_heap_lookup(gdev_vas_t *vas, uint64_t addr)
{
	gdev_mem_t *mem;
	gdev_list_t *entry = vas->memlist.next;

	while (entry) {
		mem = (gdev_mem_t *)entry->container;
		if (mem && mem->addr == addr)
			return mem;
		entry = entry->next;
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
	uint32_t sequence = ++ctx->fence.sequence[GDEV_FENCE_COMPUTE];

	/* it's important to emit a fence *after* launch():
	   the LAUNCH method of the PGRAPH engine is not associated with
	   the QUERY method, i.e., we have to submit the QUERY method 
	   explicitly after the kernel is launched. */
	compute->launch(ctx, kern);
	compute->fence_write(ctx, GDEV_FENCE_COMPUTE, sequence);
	
	return sequence;
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
void gdev_poll(gdev_ctx_t *ctx, int type, uint32_t sequence)
{
	gdev_vas_t *vas = ctx->vas;
	gdev_device_t *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t poll_times = 0;
	uint32_t val;

	compute->fence_read(ctx, type, &val);

	while (val < sequence || val > sequence + GDEV_FENCE_LIMIT) {
		/* relax the polling after some time. */
		if (poll_times > 0x80000000) {
			SCHED_YIELD();
		}
		else if (poll_times == 0xffffffff) {
			poll_times = 0;
		}
		poll_times++;
		compute->fence_read(ctx, type, &val);
	}

	/* sequence rolls back to zero, if necessary. */
	if (ctx->fence.sequence[type] == GDEV_FENCE_LIMIT) {
		ctx->fence.sequence[type] = 0;
	}
}
