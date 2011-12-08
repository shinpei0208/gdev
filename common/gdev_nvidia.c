/*
 * Copyright 2011 Shinpei Kato
 *
 * University of California at Santa Cruz
 * Systems Research Lab.
 *
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
#include "gdev_proto.h"
#include "gdev_time.h"

/* initialize the compute engine. */
int gdev_compute_init(struct gdev_device *gdev, int minor, void *priv)
{
	gdev->id = minor;
	gdev->priv = priv;
	gdev->users = 0;
	gdev->mem_used = 0;
	gdev_query(gdev, GDEV_NVIDIA_QUERY_DEVICE_MEM_SIZE, &gdev->mem_size);
	gdev_query(gdev, GDEV_NVIDIA_QUERY_CHIPSET, (uint64_t*) &gdev->chipset);
	gdev_list_init(&gdev->vas_list, NULL); /* VAS list. */

    switch (gdev->chipset & 0xf0) {
    case 0xC0:
		nvc0_compute_setup(gdev);
        break;
    case 0x50:
    case 0x80:
    case 0x90:
    case 0xA0:
		/* TODO: create the compute and m2mf subchannels! */
		GDEV_PRINT("NV%x not supported.\n", gdev->chipset);
		return -EINVAL;
    default:
		GDEV_PRINT("NV%x not supported.\n", gdev->chipset);
		return -EINVAL;
    }

	return 0;
}

void gdev_vas_list_add(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	gdev_list_add(&vas->list_entry, &gdev->vas_list);
}

/* delete the VAS object from the device VAS list. */
void gdev_vas_list_del(struct gdev_vas *vas)
{
	gdev_list_del(&vas->list_entry);
}

/* add the device memory object to the memory list. */
void gdev_mem_list_add(struct gdev_mem *mem, int type)
{
	struct gdev_vas *vas = mem->vas;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_list_add(&mem->list_entry, &vas->mem_list);
		break;
	case GDEV_MEM_DMA:
		gdev_list_add(&mem->list_entry, &vas->dma_mem_list);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
}

/* delete the device memory object from the memory list. */
void gdev_mem_list_del(struct gdev_mem *mem)
{
	gdev_list_del(&mem->list_entry);
}

/* look up the memory object allocated at the specified address. */
struct gdev_mem *gdev_mem_lookup(struct gdev_vas *vas, uint64_t addr, int type)
{
	struct gdev_mem *mem;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_list_for_each (mem, &vas->mem_list) {
			if (mem && (mem->addr == addr))
				return mem;
		}
		break;
	case GDEV_MEM_DMA:
		gdev_list_for_each (mem, &vas->dma_mem_list) {
			if (mem && (mem->map == (void *)addr))
				return mem;
		}
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}

	return NULL;
}

/* free all memory left in heap. */
void gdev_garbage_collect(struct gdev_vas *vas)
{
	struct gdev_mem *mem;

	/* device memory. */
	gdev_list_for_each (mem, &vas->mem_list) {
		gdev_mem_free(mem);
		GDEV_PRINT("Freed at 0x%x.\n", (uint32_t) GDEV_MEM_ADDR(mem));
	}

	/* host DMA memory. */
	gdev_list_for_each (mem, &vas->dma_mem_list) {
		gdev_mem_free(mem);
		GDEV_PRINT("Freed at 0x%x.\n", (uint32_t) GDEV_MEM_ADDR(mem));
	}
}

/* copy data of @size from @src_addr to @dst_addr. */
uint32_t gdev_memcpy
(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t sequence = ++ctx->fence.sequence[GDEV_FENCE_DMA];

	compute->membar(ctx);
	/* it's important to emit a fence *before* memcpy():
	   the EXEC method of the PCOPY and M2MF engines is associated with
	   the QUERY method, i.e., if QUERY is set, the sequence will be 
	   written to the specified address when the data are transfered. */
	compute->fence_write(ctx, GDEV_FENCE_DMA, sequence);
	compute->memcpy(ctx, dst_addr, src_addr, size);

	return sequence;
}

/* launch the kernel onto the GPU. */
uint32_t gdev_launch(struct gdev_ctx *ctx, struct gdev_kernel *kern)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t seq = ++ctx->fence.sequence[GDEV_FENCE_COMPUTE];

	compute->membar(ctx);
	/* it's important to emit a fence *after* launch():
	   the LAUNCH method of the PGRAPH engine is not associated with
	   the QUERY method, i.e., we have to submit the QUERY method 
	   explicitly after the kernel is launched. */
	compute->launch(ctx, kern);
	compute->fence_write(ctx, GDEV_FENCE_COMPUTE, seq);
	
	return seq;
}

/* poll until the resource becomes available. */
int gdev_poll
(struct gdev_ctx *ctx, int type, uint32_t seq, struct gdev_time *timeout)
{
	struct gdev_time time_start, time_now, time_elapse, time_relax;
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;
	uint32_t val;

	gdev_time_stamp(&time_start);
	gdev_time_ms(&time_relax, 1); /* relax polling when 1 ms elapsed. */

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
