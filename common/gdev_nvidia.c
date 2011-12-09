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

#include "gdev_api.h"
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

/* query device-specific information. */
int gdev_query(struct gdev_device *gdev, uint32_t type, uint64_t *result)
{
	return gdev_raw_query(gdev, type, result);
}

/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_dev_open(int minor)
{
	return gdev_raw_dev_open(minor);
}

/* close the specified Gdev object. */
void gdev_dev_close(struct gdev_device *gdev)
{
	gdev_raw_dev_close(gdev);
}

/* allocate a new virual address space (VAS) object. */
struct gdev_vas *gdev_vas_new(struct gdev_device *gdev, uint64_t size)
{
	struct gdev_vas *vas;

	if (!(vas = gdev_raw_vas_new(gdev, size))) {
		return NULL;
	}

	vas->gdev = gdev;
	gdev_list_init(&vas->list_entry, (void *) vas); /* entry to VAS list. */
	gdev_list_init(&vas->mem_list, NULL); /* device memory list. */
	gdev_list_init(&vas->dma_mem_list, NULL); /* host dma memory list. */

	return vas;
}

/* free the specified virtual address space object. */
void gdev_vas_free(struct gdev_vas *vas)
{
	gdev_raw_vas_free(vas);
}

/* create a new GPU context object. */
struct gdev_ctx *gdev_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	struct gdev_compute *compute = gdev->compute;

	if (!(ctx = gdev_raw_ctx_new(gdev, vas))) {
		return NULL;
	}

	ctx->vas = vas;

	/* initialize the channel. */
	compute->init(ctx);

	return ctx;
}

/* destroy the specified GPU context object. */
void gdev_ctx_free(struct gdev_ctx *ctx)
{
	gdev_raw_ctx_free(ctx);
}

/* initialize a memory object. */
static void __gdev_mem_init
(struct gdev_mem *mem, struct gdev_vas *vas, uint64_t addr, uint64_t size, 
 void *map, int type)
{
	mem->vas = vas;
	mem->addr = addr;
	mem->size = size;
	mem->map = map;
	mem->type = type;
	mem->evicted = 0;
	mem->swap.buf = NULL;
	mem->swap.addr = 0;
	mem->swap.size = 0;
	mem->swap.type = 0;
	mem->evict_info.holder = NULL;
	mem->evict_info.victim = NULL;
	mem->evict_info.criminal = NULL;

	gdev_list_init(&mem->list_entry, (void *) mem);
}

/* allocate a new memory object. */
struct gdev_mem *gdev_mem_alloc(struct gdev_vas *vas, uint64_t size, int type)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *mem;
	uint64_t addr;
	void *map;

	switch (type) {
	case GDEV_MEM_DEVICE:
		if (!(mem = gdev_raw_mem_alloc(vas, &addr, &size, &map)))
			goto fail;
		gdev->mem_used += size;
		break;
	case GDEV_MEM_DMA:
		if (!(mem = gdev_raw_mem_alloc_dma(vas, &addr, &size, &map)))
			goto fail;
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
		goto fail;
	}

	__gdev_mem_init(mem, vas, addr, size, map, type);

	return mem;

fail:
	return NULL;
}

/* free the specified memory object. */
void gdev_mem_free(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;

	if (mem->type == GDEV_MEM_DEVICE)
		gdev->mem_used -= mem->size;

	gdev_raw_mem_free(mem);
}

/* find a memory object that we can borrow some memory space from. */
static struct gdev_mem *__gdev_mem_select_victim
(struct gdev_vas *vas, uint64_t size, int type)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_vas *v;
	struct gdev_mem *m;
	struct gdev_list *mem_list_ptr;
	void *sysmem;
	uint64_t flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		mem_list_ptr = &vas->mem_list;
		break;
	case GDEV_MEM_DMA:
		mem_list_ptr = &vas->dma_mem_list;
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
		goto fail;
	}

	/* select by low priorities. */
	LOCK(&gdev->vas_lock);
	gdev_list_for_each(v, &gdev->vas_list) {
		LOCK_NESTED(&v->mem_lock, &flags);
		gdev_list_for_each(m, mem_list_ptr) {
			if (m->size >= size && !m->evicted) {
				size = m->size; /* FIXME */
				if (!(sysmem = MALLOC(size))) {
					UNLOCK_NESTED(&v->mem_lock, &flags);
					UNLOCK(&gdev->vas_lock);
					goto fail;
				}
				m->swap.buf = sysmem;
				m->swap.addr = m->addr;
				m->swap.size = size;
				m->swap.type = type;
				m->evicted = 1;
				UNLOCK_NESTED(&v->mem_lock, &flags);
				UNLOCK(&gdev->vas_lock);
				return m;
			}
		}
		UNLOCK_NESTED(&v->mem_lock, &flags);
	}
	UNLOCK(&gdev->vas_lock);

fail:
	return NULL;
}

/* evict some memory object and assign that space for the new object. */
struct gdev_mem *gdev_mem_evict
(struct gdev_vas *vas, uint64_t size, int type, void *h)
{
	struct gdev_mem *mem, *shmem;
	struct gdev_swap *swap;
	uint64_t addr;
	void *map;

	/* select a victim memory object. */
	if (!(mem = __gdev_mem_select_victim(vas, size, GDEV_MEM_DEVICE)))
		goto fail_victim;

	/* evict data. we should also zero-clear the buffer for security. */
	swap = &mem->swap;
	if (gmemcpy_from_device(h, swap->buf, swap->addr, swap->size))
		goto fail_gmemcpy;

	/* reuse the same (physical) memory space by sharing. */
	if (!(shmem = gdev_raw_mem_share(vas, mem, &addr, &size, &map)))
		goto fail_shmem;

	__gdev_mem_init(shmem, vas, addr, size, map, type);
	shmem->evict_info.victim = mem;
	mem->evict_info.holder = shmem;
	mem->evict_info.criminal = shmem;
	if (mem->evict_info.victim)
		mem->evict_info.victim->evict_info.holder = shmem;

	return shmem;

fail_shmem:
	gdev_mem_reload(mem, h);
fail_gmemcpy:
	FREE(swap->buf);
fail_victim:
	return NULL;
}

int gdev_mem_reload(struct gdev_mem *mem, void *h)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_swap *swap = &mem->swap;

	LOCK(&vas->mem_lock);
	mem->evicted = 0;
	UNLOCK(&vas->mem_lock);
	return gmemcpy_to_device(h, swap->addr, swap->buf, swap->size);
}

/* garbage collection: free all memory left in heap. */
void gdev_mem_gc(struct gdev_vas *vas)
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

void gdev_vas_list_add(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;

	LOCK(&gdev->vas_lock);
	gdev_list_add(&vas->list_entry, &gdev->vas_list);
	UNLOCK(&gdev->vas_lock);
}

/* delete the VAS object from the device VAS list. */
void gdev_vas_list_del(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;

	LOCK(&gdev->vas_lock);
	gdev_list_del(&vas->list_entry);
	UNLOCK(&gdev->vas_lock);
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
