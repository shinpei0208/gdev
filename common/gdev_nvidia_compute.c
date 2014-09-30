/*
 * Copyright (C) Shinpei Kato
 *
 * University of California, Santa Cruz
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
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gdev_api.h"
#include "gdev_device.h"

/* set up the architecture-dependent compute engine. */
int gdev_compute_setup(struct gdev_device *gdev)
{
    	if (!(gdev->chipset & 0xf000)) { 
	    switch (gdev->chipset & 0xf0) {
		case 0xE0:
		case 0xF0:
		    nve4_compute_setup(gdev);
		    break;
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
	}else{
#ifdef GDEV_DRIVER_BARRA
	    /* Override arch-dependent compute setup */
	    barra_compute_setup(gdev);
#else
	    return -EINVAL;
#endif
	}
	return 0;
}

/* launch the kernel onto the GPU. */
uint32_t gdev_launch(struct gdev_ctx *ctx, struct gdev_kernel *kern)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *dev_swap = gdev_swap_get(gdev);
	struct gdev_compute *compute = gdev_compute_get(gdev);
	uint32_t seq;

	/* evict data saved in device swap memory space to host memory. */
	if (dev_swap && dev_swap->shm->holder) {
		struct gdev_mem *mem = dev_swap->shm->holder;
		gdev_shm_evict_conflict(ctx, mem->swap_mem); /* don't use gdev->swap */
		dev_swap->shm->holder = NULL;
	}

	if (++ctx->fence.seq == GDEV_FENCE_COUNT)
		ctx->fence.seq = 1;
	seq = ctx->fence.seq;

	compute->membar(ctx);
	/* it's important to emit a fence *after* launch():
	   the LAUNCH method of the PGRAPH engine is not associated with
	   the QUERY method, i.e., we have to submit the QUERY method 
	   explicitly after the kernel is launched. */
	compute->fence_reset(ctx, seq);
	compute->launch(ctx, kern);
	compute->fence_write(ctx, GDEV_OP_COMPUTE, seq);

#ifndef GDEV_SCHED_DISABLED
	/* set an interrupt to be caused when compute done. */
	compute->notify_intr(ctx);
#endif
	
	return seq;
}

/* copy data of @size from @src_addr to @dst_addr. */
uint32_t gdev_memcpy(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev_compute_get(gdev);
	uint32_t seq;

	if (++ctx->fence.seq == GDEV_FENCE_COUNT)
		ctx->fence.seq = 1;
	seq = ctx->fence.seq;

	compute->membar(ctx);
	/* it's important to emit a fence *before* memcpy():
	   the EXEC method of the PCOPY and M2MF engines is associated with
	   the QUERY method, i.e., if QUERY is set, the sequence will be 
	   written to the specified address when the data are transfered. */
	compute->fence_reset(ctx, seq);
	if( (gdev->chipset & 0xf0) >= 0xe0 || (gdev->chipset & 0xf000) ) {
	    compute->memcpy(ctx, dst_addr, src_addr, size);
	    compute->fence_write(ctx, GDEV_OP_COMPUTE /* == COMPUTE */, seq);
	}
	else {
	    compute->fence_write(ctx, GDEV_OP_MEMCPY /* == M2MF */, seq);
	    compute->memcpy(ctx, dst_addr, src_addr, size);
	}

	return seq;
}

/* asynchronously copy data of @size from @src_addr to @dst_addr. */
uint32_t gdev_memcpy_async(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev_compute_get(gdev);
	uint32_t seq;

	if (++ctx->fence.seq == GDEV_FENCE_COUNT)
		ctx->fence.seq = 1;
	seq = ctx->fence.seq;

	compute->membar(ctx);
	/* it's important to emit a fence *before* memcpy():
	   the EXEC method of the PCOPY and M2MF engines is associated with
	   the QUERY method, i.e., if QUERY is set, the sequence will be 
	   written to the specified address when the data are transfered. */
	compute->fence_reset(ctx, seq);
	if( (gdev->chipset & 0xf0) >= 0xe0) {
	    compute->memcpy_async(ctx, dst_addr, src_addr, size);
	    compute->fence_write(ctx, GDEV_OP_COMPUTE /* == COMPUTE */, seq);
	}
	else {
	    compute->fence_write(ctx, GDEV_OP_MEMCPY_ASYNC /* == PCOPY0 */, seq);
	    compute->memcpy_async(ctx, dst_addr, src_addr, size);
	}

	return seq;
}

/* read 32-bit value from @addr. */
uint32_t gdev_read32(struct gdev_mem *mem, uint64_t addr)
{
	return gdev_raw_read32(mem, addr);
}

/* write 32-bit @val to @addr. */
void gdev_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val)
{
	gdev_raw_write32(mem, addr, val);
}

/* read @size of data from @addr. */
int gdev_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size)
{
	return gdev_raw_read(mem, buf, addr, size);
}

/* write @size of data to @addr. */
int gdev_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size)
{
	return gdev_raw_write(mem, addr, buf, size);
}

/* poll until the resource becomes available. */
int gdev_poll(struct gdev_ctx *ctx, uint32_t seq, struct gdev_time *timeout)
{
	struct gdev_time time_start, time_now, time_elapse, time_relax;
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev_compute_get(gdev);

	gdev_time_stamp(&time_start);
	gdev_time_ms(&time_relax, 100); /* relax polling when 100 ms elapsed. */

	while (seq != compute->fence_read(ctx, seq)) {
		gdev_time_stamp(&time_now);
		/* time_elapse = time_now - time_start */
		gdev_time_sub(&time_elapse, &time_now, &time_start);
		/* relax polling after some time. */
		if (gdev_time_ge(&time_elapse, &time_relax))
			SCHED_YIELD();
		/* check timeout. */
		if (timeout && gdev_time_ge(&time_elapse, timeout))
			return -ETIME;
	}

	compute->fence_reset(ctx, seq);

	return 0;
}

/* barrier memory by blocking. */
int gdev_barrier(struct gdev_ctx *ctx)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev_compute_get(gdev);
	uint32_t seq = 0; /* 0 is a special sequence for barrier. */

	compute->membar(ctx);
	compute->fence_write(ctx, GDEV_OP_COMPUTE, seq);
	while (seq != compute->fence_read(ctx, seq));

	return 0;
}

/* query device-specific information. */
int gdev_query(struct gdev_device *gdev, uint32_t type, uint64_t *result)
{
	int ret;

	switch (type) {
	case GDEV_QUERY_DEVICE_MEM_SIZE:
		if (gdev->mem_size)
			*result = gdev->mem_size;
		else if ((ret = gdev_raw_query(gdev, type, result)))
			return ret;
		break;
	case GDEV_QUERY_DMA_MEM_SIZE:
		if (gdev->dma_mem_size)
			*result = gdev->dma_mem_size;
		/* FIXME: this is valid only for PCIE. */
		else  if (gdev->chipset > 0x40)
			*result = 512 * 1024 * 1024;
		else
			*result = 64 * 1024 * 1024;
		break;
	default:
		if ((ret = gdev_raw_query(gdev, type, result)))
			return ret;
	}

	return 0;
}
