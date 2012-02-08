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

#include "gdev_device.h"

/* initialize a memory object. */
void gdev_nvidia_mem_init(struct gdev_mem *mem, struct gdev_vas *vas, uint64_t addr, uint64_t size, void *map, int type)
{
	mem->vas = vas;
	mem->addr = addr;
	mem->size = size;
	mem->map = map;
	mem->type = type;
	mem->evicted = 0;
	mem->swap_mem = NULL;
	mem->swap_buf = NULL;
	mem->shm = NULL;

	gdev_list_init(&mem->list_entry_heap, (void *) mem);
	gdev_list_init(&mem->list_entry_shm, (void *) mem);
}

/* add a new memory object to the memory list. */
void gdev_nvidia_mem_list_add(struct gdev_mem *mem, int type)
{
	struct gdev_vas *vas = mem->vas;
	unsigned long flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_add(&mem->list_entry_heap, &vas->mem_list);
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_add(&mem->list_entry_heap, &vas->dma_mem_list);
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
}

/* delete the memory object from the memory list. */
void gdev_nvidia_mem_list_del(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	unsigned long flags;
	int type = mem->type;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_del(&mem->list_entry_heap);
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_del(&mem->list_entry_heap);
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}
}

/* lock the memory object so that none can change data while tranferring.
   if there are no shared memory users, no need to lock. */
void gdev_mem_lock(struct gdev_mem *mem)
{
	if (mem->shm) {
		gdev_mutex_lock(&mem->shm->mutex);
	}
}

/* unlock the memory object so that none can change data while tranferring.
   if there are no shared memory users, no need to lock. */
void gdev_mem_unlock(struct gdev_mem *mem)
{
	if (mem->shm) {
		gdev_mutex_unlock(&mem->shm->mutex);
	}
}

/* lock all the memory objects associated with @vas. */
void gdev_mem_lock_all(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *mem;

	gdev_mutex_lock(&gdev->shm_mutex);
	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_mem_lock(mem);
	}
	gdev_mutex_unlock(&gdev->shm_mutex);
}

/* unlock all the memory objects associated with @vas. */
void gdev_mem_unlock_all(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *mem;

	gdev_mutex_lock(&gdev->shm_mutex);
	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_mem_unlock(mem);
	}
	gdev_mutex_unlock(&gdev->shm_mutex);
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

	gdev_nvidia_mem_init(mem, vas, addr, size, map, type);
	gdev_nvidia_mem_list_add(mem, type);

	return mem;

fail:
	return NULL;
}

/* share memory space with @mem. if @mem is null, find a victim instead. */
struct gdev_mem *gdev_mem_share(struct gdev_vas *vas, uint64_t size)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *new;

	/* request share memory with any memory object. */
	gdev_mutex_lock(&gdev->shm_mutex);
	if (!(new = gdev_shm_attach(vas, NULL, size)))
		goto fail;
	gdev_mutex_unlock(&gdev->shm_mutex);

	return new;

fail:
	gdev_mutex_unlock(&gdev->shm_mutex);
	return NULL;
}

/* free the specified memory object. */
void gdev_mem_free(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;

	/* if the memory object is associated with shared memory, detach the 
	   shared memory. note that the memory object will be freed if users
	   become zero.
	   free the memroy object otherwise. */
	gdev_mutex_lock(&gdev->shm_mutex);
	if (mem->shm)
		gdev_shm_detach(mem);
	else {
		gdev_nvidia_mem_list_del(mem);
		gdev_raw_mem_free(mem);
	}
	gdev_mutex_unlock(&gdev->shm_mutex);
}

/* garbage collection: free all memory left in heap. */
void gdev_mem_gc(struct gdev_vas *vas)
{
#ifdef GDEV_GARBAGE_COLLECTION_SUPPORT
	struct gdev_mem *mem;

	/* device memory. */
	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		GDEV_PRINT("Garbage Collect 0x%llx\n", mem->addr);
		gdev_mem_free(mem);
	}

	/* host DMA memory. */
	gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
		gdev_mem_free(mem);
	}
#endif
}

/* look up the memory object allocated at the specified address. */
struct gdev_mem *gdev_mem_lookup(struct gdev_vas *vas, uint64_t addr, int type)
{
	struct gdev_mem *mem = NULL;
	unsigned long flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
			if (mem && (addr >= mem->addr && addr < mem->addr + mem->size))
				break;
		}
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
			uint64_t map_addr = (uint64_t) mem->map;
			if (mem && (addr >= map_addr && addr < map_addr + mem->size))
				break;
		}
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}

	return mem;
}

/* get host DMA buffer (could be memory-mapped buffer for device memory). */
void *gdev_mem_get_buf(struct gdev_mem *mem)
{
	return mem->map;
}

/* get virtual memory address. */
uint64_t gdev_mem_get_addr(struct gdev_mem *mem)
{
	return mem->addr;
}

/* get allocated memory size. */
uint64_t gdev_mem_get_size(struct gdev_mem *mem)
{
	return mem->size;
}

