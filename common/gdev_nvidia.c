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
	gdev_query(gdev, GDEV_NVIDIA_QUERY_DEVICE_MEM_SIZE, &gdev->mem_size);
	gdev_query(gdev, GDEV_NVIDIA_QUERY_CHIPSET, (uint64_t*) &gdev->chipset);
	gdev_list_init(&gdev->vas_list, NULL); /* VAS list. */
	LOCK_INIT(&gdev->lock);

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
	int ret;

	if ((ret = gdev_raw_query(gdev, type, result)))
		return ret;

	switch (type) {
	case GDEV_NVIDIA_QUERY_DEVICE_MEM_SIZE:
		*result -= 0xc010000; /* FIXME: this shouldn't be hardcoded. */
		break;
	}

	return 0;
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

	vas->prio = GDEV_PRIO_DEFAULT;
	vas->gdev = gdev;
	gdev_list_init(&vas->list_entry, (void *) vas); /* entry to VAS list. */
	gdev_list_init(&vas->mem_list, NULL); /* device memory list. */
	gdev_list_init(&vas->dma_mem_list, NULL); /* host dma memory list. */
	LOCK_INIT(&vas->lock);

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
	mem->shmem = NULL;

	gdev_list_init(&mem->list_entry_heap, (void *) mem);
	gdev_list_init(&mem->list_entry_shmem, (void *) mem);
}

/* find a memory object that we can borrow some memory space from. */
static struct gdev_mem *__gdev_mem_select_victim
(struct gdev_vas *vas, uint64_t size, int type)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_vas *v;
	struct gdev_mem *m, *victim = NULL;
	struct gdev_shmem *shmem;
	struct gdev_list *mem_list_ptr;
	unsigned long flags;

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

	if (!(shmem = MALLOC(sizeof(*shmem))))
		goto fail;

	/* select the lowest-priority object. */
	LOCK_SAVE(&gdev->lock, &flags);
	gdev_list_for_each (v, &gdev->vas_list, list_entry) {
		if (type == GDEV_MEM_DEVICE)
			mem_list_ptr = &v->mem_list;
		else if (type == GDEV_MEM_DMA)
			mem_list_ptr = &v->dma_mem_list;
		else
			break;
		LOCK_NESTED(&v->lock);
		gdev_list_for_each (m, mem_list_ptr, list_entry_heap) {
			if (m->size >= size && m->vas != vas) {
				if (!victim)
					victim = m;
				else {
					int victim_prio, m_prio;
					if (victim->shmem)
						victim_prio = victim->shmem->prio;
					else
						victim_prio = victim->vas->prio;
					if (m->shmem)
						m_prio = m->shmem->prio;
					else
						m_prio = m->vas->prio;
					if (victim_prio > m_prio)
						victim = m;
				}
			}
		}
		UNLOCK_NESTED(&v->lock);
	}

	if (!victim) {
		UNLOCK_RESTORE(&gdev->lock, &flags);
		FREE(shmem);
	}
	/* if the victim object doesn't have a shared memory object yet, 
	   allocate a new one here. */
	else if (!victim->shmem) {
		gdev_list_init(&shmem->shmem_list, NULL);
		shmem->prio = GDEV_PRIO_MIN;
		shmem->users = 0;
		shmem->holder = NULL;
		shmem->bo = victim->bo;
		victim->shmem = shmem;
		LOCK_INIT(&shmem->lock);
		LOCK_NESTED(&shmem->lock);
		/* the victim object itself must be inserted into the list. */
		gdev_list_add(&victim->list_entry_shmem, &shmem->shmem_list);
		UNLOCK_NESTED(&shmem->lock);
		victim->shmem->users++;
		UNLOCK_RESTORE(&gdev->lock, &flags);
	}
	else {
		victim->shmem->users++;
		UNLOCK_RESTORE(&gdev->lock, &flags);
		FREE(shmem); /* shmem unused. */
	}

	return victim;

fail:
	return NULL;
}

/* borrow memory space from some low-priority object. */
static struct gdev_mem *__gdev_mem_borrow
(struct gdev_vas *vas, uint64_t size, int type)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *victim, *new;
	unsigned long flags;
	uint64_t addr;
	void *map;

	/* select a victim memory object. victim->shmem will be newly allocated 
	   if NULL. shmem->users will also be incremented in advance. */
	if (!(victim = __gdev_mem_select_victim(vas, size, GDEV_MEM_DEVICE)))
		goto fail_victim;

	/* borrow the same (physical) memory space by sharing. */
	if (!(new = gdev_raw_mem_share(vas, victim, &addr, &size, &map)))
		goto fail_shmem;

	__gdev_mem_init(new, vas, addr, size, map, type);

	/* update only the master's share information. */
	LOCK_SAVE(&gdev->lock, &flags);
	new->shmem = victim->shmem;
	new->shmem->holder = new;
	if (new->shmem->prio < vas->prio)
		new->shmem->prio = vas->prio;
	/* insert the new memory object into the shared memory list. */
	LOCK_NESTED(&new->shmem->lock);
	gdev_list_add(&new->list_entry_shmem, &new->shmem->shmem_list);
	UNLOCK_NESTED(&new->shmem->lock);
	UNLOCK_RESTORE(&gdev->lock, &flags);

	GDEV_PRINT("Shared memory at 0x%llx.\n", (long long unsigned int) addr);
	return new;

fail_shmem:
	victim->shmem->users--;
fail_victim:
	return NULL;
}

/* allocate a new memory object. */
struct gdev_mem *gdev_mem_alloc(struct gdev_vas *vas, uint64_t size, int type)
{
	struct gdev_mem *mem;
	uint64_t addr;
	void *map;

	switch (type) {
	case GDEV_MEM_DEVICE:
		if (!(mem = gdev_raw_mem_alloc(vas, &addr, &size, &map)))
			goto fail_alloc;
		break;
	case GDEV_MEM_DMA:
		if (!(mem = gdev_raw_mem_alloc_dma(vas, &addr, &size, &map)))
			goto fail_alloc;
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
		goto fail;
	}

	__gdev_mem_init(mem, vas, addr, size, map, type);
	return mem;

fail_alloc:
	/* second chance by borrowing memory space. */
	if (!(mem = __gdev_mem_borrow(vas, size, type)))
		goto fail;
	return mem;

fail:
	return NULL;
}

/* free the specified memory object. */
void gdev_mem_free(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;

	LOCK_SAVE(&gdev->lock, &flags);
	/* if the memory object is not shared, free it. */
	if (!mem->shmem) {
		UNLOCK_RESTORE(&gdev->lock, &flags);
		gdev_raw_mem_free(mem);
	}
	/* if the memory object is shared but no users, free it. */
	else if (mem->shmem->users == 0) {
		UNLOCK_RESTORE(&gdev->lock, &flags);
		FREE(mem->shmem);
		gdev_raw_mem_free(mem);
	}
	/* otherwise, just unshare the memory object. */
	else {
		struct gdev_mem *m;
		LOCK_NESTED(&mem->shmem->lock);
		gdev_list_del(&mem->list_entry_shmem);
		/* if a freeing memory object has the highest priority among the 
		   shared memory objects, find the next highest priority one. */
		if (mem->shmem->prio == vas->prio) {
			int prio = GDEV_PRIO_MIN;
			gdev_list_for_each (m, &mem->shmem->shmem_list, list_entry_shmem) {
				if (m->vas->prio < prio)
					prio = m->vas->prio;
			}
			mem->shmem->prio = prio;
		}
		if (mem->shmem->holder == mem)
			mem->shmem->holder = NULL;
		mem->shmem->users--;
		UNLOCK_NESTED(&mem->shmem->lock);
		UNLOCK_RESTORE(&gdev->lock, &flags);
		gdev_raw_mem_unshare(mem);
	}
}

/* evict the memory object data. */
int gdev_mem_evict(struct gdev_mem *mem, void *h)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_swap *swap;
	unsigned long flags;
	uint64_t size;
	void *sysmem;
	int ret;

	size = mem->size; /* FIXME */
	if (!(sysmem = MALLOC(size))) {
		ret = -ENOMEM;
		goto fail_malloc;
	}

	LOCK_SAVE(&gdev->lock, &flags);
	mem->swap.buf = sysmem;
	mem->swap.addr = mem->addr;
	mem->swap.size = size;
	mem->swap.type = mem->type;
	mem->evicted = 1;
	UNLOCK_RESTORE(&gdev->lock, &flags);

	swap = &mem->swap;
	if ((ret = gmemcpy_from_device(h, swap->buf, swap->addr, swap->size)))
		goto fail_gmemcpy;
	
	return 0;

	
fail_gmemcpy:
	FREE(swap->buf);
fail_malloc:
	return ret;
}

int gdev_mem_reload(struct gdev_mem *mem, void *h)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_swap *swap = &mem->swap;
	unsigned long flags;
	int ret;

	if ((ret = gmemcpy_to_device(h, swap->addr, swap->buf, swap->size)))
		goto fail_gmemcpy;

	LOCK_SAVE(&vas->lock, &flags);
	mem->evicted = 0;
	UNLOCK_RESTORE(&vas->lock, &flags);

fail_gmemcpy:
	return ret;
}

/* garbage collection: free all memory left in heap. */
void gdev_mem_gc(struct gdev_vas *vas)
{
	struct gdev_mem *mem;

	/* device memory. */
	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_mem_free(mem);
	}

	/* host DMA memory. */
	gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
		gdev_mem_free(mem);
	}
}

void gdev_vas_list_add(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;

	LOCK_SAVE(&gdev->lock, &flags);
	gdev_list_add(&vas->list_entry, &gdev->vas_list);
	UNLOCK_RESTORE(&gdev->lock, &flags);
}

/* delete the VAS object from the device VAS list. */
void gdev_vas_list_del(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;

	LOCK_SAVE(&gdev->lock, &flags);
	gdev_list_del(&vas->list_entry);
	UNLOCK_RESTORE(&gdev->lock, &flags);
}

/* add the device memory object to the memory list. */
void gdev_mem_list_add(struct gdev_mem *mem, int type)
{
	struct gdev_vas *vas = mem->vas;
	unsigned long flags;

	LOCK_SAVE(&vas->lock, &flags);
	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_list_add(&mem->list_entry_heap, &vas->mem_list);
		break;
	case GDEV_MEM_DMA:
		gdev_list_add(&mem->list_entry_heap, &vas->dma_mem_list);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
	UNLOCK_RESTORE(&vas->lock, &flags);
}

/* delete the device memory object from the memory list. */
void gdev_mem_list_del(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	int type = mem->type;
	unsigned long flags;

	LOCK_SAVE(&vas->lock, &flags);
	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_list_del(&mem->list_entry_heap);
		break;
	case GDEV_MEM_DMA:
		gdev_list_del(&mem->list_entry_heap);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
	UNLOCK_RESTORE(&vas->lock, &flags);
}

/* look up the memory object allocated at the specified address. */
struct gdev_mem *gdev_mem_lookup(struct gdev_vas *vas, uint64_t addr, int type)
{
	struct gdev_mem *mem = NULL;
	unsigned long flags;

	LOCK_SAVE(&vas->lock, &flags);
	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
			if (mem && (mem->addr == addr))
				break;
		}
		break;
	case GDEV_MEM_DMA:
		gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
			if (mem && (mem->map == (void *)addr))
				break;
		}
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
	UNLOCK_RESTORE(&vas->lock, &flags);

	return mem;
}
