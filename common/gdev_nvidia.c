/*
 * Copyright 2011 Shinpei Kato
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

/**
 * these functions must be used when evicting/reloading data.
 * better implementations wanted!
 */
int gdev_callback_evict_to_host(void*, void*, uint64_t, uint64_t);
int gdev_callback_evict_to_device(void*, uint64_t, uint64_t, uint64_t);
int gdev_callback_reload_from_host(void*, uint64_t, void*, uint64_t);
int gdev_callback_reload_from_device(void*, uint64_t, uint64_t, uint64_t);

/**
 * memcpy functions prototypes
 */
static uint32_t gdev_memcpy_sync
(struct gdev_ctx *, uint64_t, uint64_t, uint32_t, uint32_t);
static uint32_t gdev_memcpy_async
(struct gdev_ctx *, uint64_t, uint64_t, uint32_t, uint32_t);

/**
 * pointers to memcpy functions.
 * gdev_memcpy_func[0] is synchronous memcpy.
 * gdev_memcpy_func[1] is asynchrounous memcpy.
 */
static uint32_t (*gdev_memcpy_func[2])
(struct gdev_ctx*, uint64_t, uint64_t, uint32_t, uint32_t) = {
	gdev_memcpy_sync,
	gdev_memcpy_async
};

/* set up the architecture-dependent compute engine. */
int gdev_compute_setup(struct gdev_device *gdev)
{
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
	struct gdev_mem *dev_swap = gdev->swap;
	struct gdev_compute *compute = gdev->compute;
	uint32_t seq;

	/* evict data saved in device swap memory space to host memory. */
	if (dev_swap && dev_swap->shmem->holder) {
		struct gdev_mem *mem = dev_swap->shmem->holder;
		gdev_shmem_evict(ctx, mem->swap_mem); /* don't pass gdev->swap! */
		dev_swap->shmem->holder = NULL;
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
	compute->fence_write(ctx, GDEV_SUBCH_COMPUTE, seq);

	/* set an interrupt to be caused when compute done. */
	compute->notify_intr(ctx);
	
	return seq;
}

/* synchrounously copy data of @size from @src_addr to @dst_addr. */
static uint32_t gdev_memcpy_sync
(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size, 
 uint32_t seq)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;

	compute->membar(ctx);
	/* it's important to emit a fence *before* memcpy():
	   the EXEC method of the PCOPY and M2MF engines is associated with
	   the QUERY method, i.e., if QUERY is set, the sequence will be 
	   written to the specified address when the data are transfered. */
	compute->fence_reset(ctx, seq);
	compute->fence_write(ctx, GDEV_SUBCH_M2MF, seq);
	compute->memcpy(ctx, dst_addr, src_addr, size);

	return seq;
}

/* asynchrounously copy data of @size from @src_addr to @dst_addr. */
static uint32_t gdev_memcpy_async
(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size, 
 uint32_t seq)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_compute *compute = gdev->compute;

	compute->membar(ctx);
	/* it's important to emit a fence *before* memcpy():
	   the EXEC method of the PCOPY and M2MF engines is associated with
	   the QUERY method, i.e., if QUERY is set, the sequence will be 
	   written to the specified address when the data are transfered. */
	compute->fence_reset(ctx, seq);
	compute->fence_write(ctx, GDEV_SUBCH_PCOPY0, seq);
	compute->memcpy_async(ctx, dst_addr, src_addr, size);

	return seq;
}

/* asynchrounously copy data of @size from @src_addr to @dst_addr. */
uint32_t gdev_memcpy
(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size, 
 int async)
{
	uint32_t seq;

	if (++ctx->fence.seq == GDEV_FENCE_COUNT)
		ctx->fence.seq = 1;
	seq = ctx->fence.seq;

	return gdev_memcpy_func[async](ctx, dst_addr, src_addr, size, seq);
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
	struct gdev_compute *compute = gdev->compute;

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
struct gdev_vas *gdev_vas_new(struct gdev_device *gdev, uint64_t size, void *h)
{
	struct gdev_vas *vas;

	if (!(vas = gdev_raw_vas_new(gdev, size))) {
		return NULL;
	}

	vas->handle = h;
	vas->gdev = gdev;
	vas->prio = GDEV_PRIO_DEFAULT;
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
	mem->swap_mem = NULL;
	mem->swap_buf = NULL;
	mem->shmem = NULL;

	gdev_list_init(&mem->list_entry_heap, (void *) mem);
	gdev_list_init(&mem->list_entry_shmem, (void *) mem);
}

/* attach device memory and allocate host buffer for swap */
static int __gdev_swap_create(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *swap_mem;
	uint64_t addr, size;
	void *map, *swap_buf;

	/* host buffer for swap. */
	swap_buf = MALLOC(mem->size);
	if (!swap_buf)
		goto fail_swap_buf;
	/* device memory for temporal swap (shared by others). */
	if (GDEV_SWAP_MEM_SIZE > 0) {
		swap_mem = gdev_raw_mem_share(vas, gdev->swap, &addr, &size, &map);
		if (!swap_mem)
			goto fail_swap_mem;
		__gdev_mem_init(swap_mem, vas, addr, size, map,	GDEV_MEM_DEVICE);
	}
	else {
		swap_mem = NULL;
	}
	mem->swap_buf = swap_buf;
	mem->swap_mem = swap_mem;

	return 0;

fail_swap_mem:
	FREE(swap_buf);
fail_swap_buf:
	return -ENOMEM;
}

/* dettach device memory and free host buffer for swap */
static void __gdev_swap_destroy(struct gdev_mem *mem)
{
	if (GDEV_SWAP_MEM_SIZE > 0)
		gdev_raw_mem_unshare(mem->swap_mem);
	FREE(mem->swap_buf);
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
			goto fail;
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
	struct gdev_shmem *shmem = mem->shmem;
	struct gdev_mem *m;

	MUTEX_LOCK(&gdev->shmem_mutex); /* be careful of dead lock. */

	/* if the memory object is not shared, free it. */
	if (!shmem) {
		gdev_raw_mem_free(mem);
	}
	else {
		MUTEX_LOCK(&shmem->mutex);
		mem->shmem = NULL;
		gdev_list_del(&mem->list_entry_shmem);
		__gdev_swap_destroy(mem);
		/* if the memory object is shared but no users, free it. 
		   since users == 0, no one else will use mem->shmem. */
		if (shmem->users == 0) {
			gdev_raw_mem_free(mem);
			MUTEX_UNLOCK(&shmem->mutex);
			FREE(shmem);
		}
		/* otherwise, just unshare the memory object. */
		else {
			/* if a freeing memory object has the highest priority among the 
			   shared memory objects, find the next highest priority one. */
			if (shmem->prio == vas->prio) {
				int prio = GDEV_PRIO_MIN;
				gdev_list_for_each (m, &shmem->shmem_list, list_entry_shmem) {
					if (m->vas->prio < prio)
						prio = m->vas->prio;
				}
				shmem->prio = prio;
			}
			if (shmem->holder == mem)
				shmem->holder = NULL;
			/* unshare the memory object. */
			gdev_raw_mem_unshare(mem);
			shmem->users--;
			MUTEX_UNLOCK(&shmem->mutex);
		}
	}

	MUTEX_UNLOCK(&gdev->shmem_mutex);
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

/* evict the shared memory object data.
   the shared memory object associated with @mem must be locked. */
int gdev_shmem_evict(struct gdev_ctx *ctx, struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *holder;
	uint64_t src_addr;
	uint64_t size;
	int ret;

	if (mem->shmem) {
		void *h = ctx->vas->handle;
		if (mem->shmem->holder && mem->shmem->holder != mem) {
			struct gdev_mem *dev_swap = gdev->swap;
			holder = mem->shmem->holder;
			src_addr = mem->addr;
			size = mem->shmem->size;
			if (dev_swap && !dev_swap->shmem->holder) {
				uint64_t dst_addr = holder->swap_mem->addr;
				ret = gdev_callback_evict_to_device(h,dst_addr,src_addr,size);
				if (ret)
					goto fail_evict;
				dev_swap->shmem->holder = mem;
			}
			else {
				void *dst_buf = holder->swap_buf;
				ret = gdev_callback_evict_to_host(h,dst_buf,src_addr,size);
				if (ret)
					goto fail_evict;
			}
			holder->evicted = 1;
		}
		mem->shmem->holder = mem;
	}

	return 0;

	
fail_evict:
	return ret;
}

/* evict all the shared memory object data associated to @vas. 
   all the shared memory objects associated to @vas must be locked. */
int gdev_shmem_evict_all(struct gdev_ctx *ctx, struct gdev_vas *vas)
{
	struct gdev_mem *mem;

	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_shmem_evict(ctx, mem);
	}

	return 0;
}

/* reload the evicted memory object data. 
   the shared memory object associated with @mem must be locked. */
int gdev_shmem_reload(struct gdev_ctx *ctx, struct gdev_mem *mem)
{
	struct gdev_vas *vas = ctx->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *dev_swap;
	uint64_t dst_addr;
	uint64_t size;
	int ret;

	if (mem->evicted) {
		void *h = vas->handle;
		/* evict the corresponding memory space first. */
		gdev_shmem_evict(ctx, mem);
		/* reload data regardless whether eviction succeeded or failed. */
		dev_swap = gdev->swap;
		dst_addr = mem->addr;
		size = mem->size;
		if (dev_swap && dev_swap->shmem->holder == mem) {
			uint64_t src_addr = mem->swap_mem->addr;
			ret = gdev_callback_reload_from_device(h, dst_addr, src_addr, size);
			if (ret)
				goto fail_reload;
		}
		else {
			void *src_buf = mem->swap_buf;
			ret = gdev_callback_reload_from_host(h, dst_addr, src_buf, size);
			if (ret)
				goto fail_reload;
		}
		mem->evicted = 0;
		mem->shmem->holder = mem;
	}

	return 0;

fail_reload:
	return ret;
}

/* reload all the evicted memory object data associated to @vas.
   all the shared memory objects associated to @vas must be locked. */
int gdev_shmem_reload_all(struct gdev_ctx *ctx, struct gdev_vas *vas)
{
	struct gdev_mem *mem;

	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_shmem_reload(ctx, mem);
	}

	return 0;
}

/* find a memory object that we can borrow some memory space from. */
static struct gdev_mem *__gdev_shmem_find_victim
(struct gdev_vas *vas, uint64_t size)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_vas *v;
	struct gdev_mem *m, *victim = NULL;
	struct gdev_shmem *shmem;
	unsigned long flags;

	if (!(shmem = MALLOC(sizeof(*shmem))))
		goto fail_shmem;

	/* be careful of dead lock. this lock is valid only for shared memory. */
	MUTEX_LOCK(&gdev->shmem_mutex);

	/* select the lowest-priority object. */
	LOCK_SAVE(&gdev->vas_lock, &flags);
	gdev_list_for_each (v, &gdev->vas_list, list_entry) {
		LOCK_NESTED(&v->lock);
		gdev_list_for_each (m, &v->mem_list, list_entry_heap) {
			/* don't select from the save VAS object! */
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
	UNLOCK_RESTORE(&gdev->vas_lock, &flags);

	if (!victim) {
		FREE(shmem);
	}
	/* if the victim object doesn't have a shared memory object yet, 
	   allocate a new one here. */
	else if (!victim->shmem) {
		gdev_list_init(&shmem->shmem_list, NULL);
		MUTEX_INIT(&shmem->mutex);
		shmem->prio = GDEV_PRIO_MIN;
		shmem->users = 0;
		shmem->holder = NULL;
		shmem->bo = victim->bo;
		shmem->size = victim->size;
		/* create swap for evicting data from shared memory space. */
		if (__gdev_swap_create(victim))
			goto fail_swap;
		MUTEX_LOCK(&shmem->mutex);
		/* the victim object itself must be inserted into the list. */
		gdev_list_add(&victim->list_entry_shmem, &shmem->shmem_list);
		/* someone will be using this shared memory, that's why this 
		   function was called. */
		shmem->users++;
		victim->shmem = shmem;
		MUTEX_UNLOCK(&shmem->mutex);
	}
	else {
		MUTEX_LOCK(&victim->shmem->mutex);
		victim->shmem->users++;
		MUTEX_UNLOCK(&victim->shmem->mutex);
		FREE(shmem); /* shmem was unused. */
	}

	MUTEX_UNLOCK(&gdev->shmem_mutex);

	return victim;

fail_swap:
	FREE(shmem);
	MUTEX_UNLOCK(&gdev->shmem_mutex);
fail_shmem:
	return NULL;
}

/* share memory space with @mem. if @mem is null, find a victim instead. */
struct gdev_mem *gdev_shmem_request
(struct gdev_vas *vas, struct gdev_mem *mem, uint64_t size)
{
	struct gdev_mem *new;
	uint64_t addr;
	void *map;

	if (!mem) {
		/* select a victim memory object. victim->shmem will be newly 
		   allocated if NULL, with shmem->users being incremented. */
		if (!(mem = __gdev_shmem_find_victim(vas, size)))
			goto fail_victim;
	}

	/* borrow the same (physical) memory space by sharing. */
	if (!(new = gdev_raw_mem_share(vas, mem, &addr, &size, &map)))
		goto fail_shmem;

	/* initialize the new memory object. */
	__gdev_mem_init(new, vas, addr, size, map, GDEV_MEM_DEVICE);

	/* create swap. */
	if (__gdev_swap_create(new))
		goto fail_swap;

	MUTEX_LOCK(&mem->shmem->mutex);
	new->shmem = mem->shmem;
	/* set the highest prio among users to the shared memory's priority. */
	if (new->shmem->prio < vas->prio)
		new->shmem->prio = vas->prio;
	/* insert the new memory object into the shared memory list. */
	gdev_list_add(&new->list_entry_shmem, &new->shmem->shmem_list);
	MUTEX_UNLOCK(&mem->shmem->mutex);

	return new;

fail_swap:
	gdev_raw_mem_unshare(new);
fail_shmem:
	mem->shmem->users--; /* not really needed? */
fail_victim:
	return NULL;
}

/* lock the shared memory associated with @mem, if any. */
void gdev_shmem_lock(struct gdev_mem *mem)
{
	if (mem->shmem) {
		MUTEX_LOCK(&mem->shmem->mutex);
	}
}

/* unlock the shared memory associated with @mem, if any. */
void gdev_shmem_unlock(struct gdev_mem *mem)
{
	if (mem->shmem) {
		MUTEX_UNLOCK(&mem->shmem->mutex);
	}
}

/* lock all the shared memory objects associated with @vas. */
void gdev_shmem_lock_all(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *mem;

	MUTEX_LOCK(&gdev->shmem_mutex);
	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_shmem_lock(mem);
	}
	MUTEX_UNLOCK(&gdev->shmem_mutex);
}

/* unlock all the shared memory objects associated with @vas. */
void gdev_shmem_unlock_all(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *mem;

	MUTEX_LOCK(&gdev->shmem_mutex);
	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_shmem_unlock(mem);
	}
	MUTEX_UNLOCK(&gdev->shmem_mutex);
}

/* create swap memory object for the device. */
int gdev_swap_create(struct gdev_device *gdev, uint32_t size)
{
	struct gdev_mem *swap;
	struct gdev_shmem *shmem;
	int ret = 0;

	swap = gdev_raw_swap_alloc(gdev, size);
	if (swap) {
		shmem = MALLOC(sizeof(*shmem));
		if (shmem)
			shmem->holder = NULL;
		else {
			gdev_raw_swap_free(swap);
			swap = NULL;
			ret = -ENOMEM;
		}
		swap->shmem = shmem;
	}
	else {
		ret = -ENOMEM;
	}

	gdev->swap = swap;

	return ret;
}

/* remove swap memory object from the device. */
void gdev_swap_destroy(struct gdev_device *gdev)
{
	if (gdev->swap) {
		if (gdev->swap->shmem)
			FREE(gdev->swap->shmem);
		gdev_raw_swap_free(gdev->swap);
	}
}

/* add a new VAS object into the device VAS list. */
void gdev_vas_list_add(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;
	
	LOCK_SAVE(&gdev->vas_lock, &flags);
	gdev_list_add(&vas->list_entry, &gdev->vas_list);
	UNLOCK_RESTORE(&gdev->vas_lock, &flags);
}

/* delete the VAS object from the device VAS list. */
void gdev_vas_list_del(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;
	
	LOCK_SAVE(&gdev->vas_lock, &flags);
	gdev_list_del(&vas->list_entry);
	UNLOCK_RESTORE(&gdev->vas_lock, &flags);
}

/* add a new memory object to the memory list. */
void gdev_mem_list_add(struct gdev_mem *mem, int type)
{
	struct gdev_vas *vas = mem->vas;
	unsigned long flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		LOCK_SAVE(&vas->lock, &flags);
		gdev_list_add(&mem->list_entry_heap, &vas->mem_list);
		UNLOCK_RESTORE(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		LOCK_SAVE(&vas->lock, &flags);
		gdev_list_add(&mem->list_entry_heap, &vas->dma_mem_list);
		UNLOCK_RESTORE(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
}

/* delete the memory object from the memory list. */
void gdev_mem_list_del(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	unsigned long flags;
	int type = mem->type;

	switch (type) {
	case GDEV_MEM_DEVICE:
		LOCK_SAVE(&vas->lock, &flags);
		gdev_list_del(&mem->list_entry_heap);
		UNLOCK_RESTORE(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		LOCK_SAVE(&vas->lock, &flags);
		gdev_list_del(&mem->list_entry_heap);
		UNLOCK_RESTORE(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
}

/* look up the memory object allocated at the specified address. */
struct gdev_mem *gdev_mem_lookup(struct gdev_vas *vas, uint64_t addr, int type)
{
	struct gdev_mem *mem = NULL;
	unsigned long flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		LOCK_SAVE(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
			if (mem && (addr >= mem->addr && addr < mem->addr + mem->size))
				break;
		}
		UNLOCK_RESTORE(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		LOCK_SAVE(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
			if (mem && (addr >= (uint64_t) mem->map && 
						addr < (uint64_t) mem->map + mem->size))
				break;
		}
		UNLOCK_RESTORE(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}

	return mem;
}
