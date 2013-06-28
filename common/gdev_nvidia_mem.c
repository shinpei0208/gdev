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

#include "gdev_device.h"

/* initialize a memory object. */
void gdev_nvidia_mem_setup(struct gdev_mem *mem, struct gdev_vas *vas, int type)
{
	mem->vas = vas;
	mem->type = type;
	mem->evicted = 0;
	mem->swap_mem = NULL;
	mem->swap_buf = NULL;
	mem->shm = NULL;
	mem->map_users = 0;
	
	gdev_list_init(&mem->list_entry_heap, (void *)mem);
	gdev_list_init(&mem->list_entry_shm, (void *)mem);
}

/* add a new memory object to the memory list. */
void gdev_nvidia_mem_list_add(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	int type = mem->type;
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
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *mem;

	switch (type) {
	case GDEV_MEM_DEVICE:
		if (!(mem = gdev_raw_mem_alloc(vas, size)))
			goto fail;
		break;
	case GDEV_MEM_DMA:
		if (!(mem = gdev_raw_mem_alloc_dma(vas, size)))
			goto fail;
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
		goto fail;
	}

	gdev_nvidia_mem_setup(mem, vas, type);
	gdev_nvidia_mem_list_add(mem);

	/* update the size of memory used on the gdev device; mem->size
	 * could have been rounded up */
	gdev_mutex_lock(&gdev->shm_mutex);
	if (type == GDEV_MEM_DEVICE) {
		gdev->mem_used += mem->size;
	}
	else {
		gdev->dma_mem_used += mem->size;
	}
	gdev_mutex_unlock(&gdev->shm_mutex);

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

	/* gdev->mem_used does not have to be updated, since the size of
	 * available device memory did not change after a memory sharing */

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
	int mem_size_freed = mem->size;
	int mem_type = mem->type;

	/* if the memory object is associated with shared memory, detach the 
	   shared memory. note that the memory object will be freed if users
	   become zero.
	   free the memroy object otherwise. */
	gdev_mutex_lock(&gdev->shm_mutex);
	if (mem->shm) {
		/* if # of shm users > 1, the buffer object won't be freed yet,
		 * so the size of available device memory won't change */
		if(mem_type == GDEV_MEM_DEVICE && mem->shm->users > 1)
			mem_size_freed = 0;
		gdev_shm_detach(mem);
	}
	else {
		gdev_nvidia_mem_list_del(mem);
		gdev_raw_mem_free(mem);
	}

	if(mem_type == GDEV_MEM_DEVICE)
		gdev->mem_used -= mem_size_freed;
	else
		gdev->dma_mem_used -= mem_size_freed;
	gdev_mutex_unlock(&gdev->shm_mutex);
}

/* garbage collection: free all memory left in heap. */
void gdev_mem_gc(struct gdev_vas *vas)
{
	struct gdev_mem *mem;

	/* device memory. */
	while((mem = gdev_list_container(gdev_list_head(&vas->mem_list)))) {
		gdev_mem_free(mem);
	}

	/* host DMA memory. */
	while((mem = gdev_list_container(gdev_list_head(&vas->dma_mem_list)))) {
		gdev_mem_free(mem);
	}
}

/* map device memory to host DMA memory. */
void *gdev_mem_map(struct gdev_mem *mem, uint64_t offset, uint64_t size)
{
	if (offset + size > mem->size)
		return NULL;

	/* @size is not really used here... */
	if (mem->map_users == 0 && mem->size > GDEV_MEM_MAPPABLE_LIMIT) {
		mem->map = gdev_raw_mem_map(mem);
		if (!mem->map)
			return NULL;
	}

	mem->map_users++;
	return mem->map + offset;
}

/* unmap device memory from host DMA memory. */
void gdev_mem_unmap(struct gdev_mem *mem)
{
	mem->map_users--;
	if (mem->map_users == 0 && mem->size > GDEV_MEM_MAPPABLE_LIMIT) {
		gdev_raw_mem_unmap(mem, mem->map);
		mem->map = NULL;
	}
}

/* look up a memory object associated with device virtual memory address. */
struct gdev_mem *gdev_mem_lookup_by_addr(struct gdev_vas *vas, uint64_t addr, int type)
{
	struct gdev_mem *mem = NULL;
	unsigned long flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
			if ((addr >= mem->addr) && (addr < mem->addr + mem->size))
				break;
		}
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
			if ((addr >= mem->addr) && (addr < mem->addr + mem->size))
				break;
		}
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	default:
		GDEV_PRINT("Memory type not supported\n");
	}

	return mem;
}

/* look up a memory object associated with host buffer address. */
struct gdev_mem *gdev_mem_lookup_by_buf(struct gdev_vas *vas, const void *buf, int type)
{
	struct gdev_mem *mem = NULL;
	uint64_t addr = (uint64_t)buf;
	unsigned long flags;

	switch (type) {
	case GDEV_MEM_DEVICE:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
			uint64_t map_addr = (uint64_t)mem->map;
			if ((addr >= map_addr) && (addr < map_addr + mem->size))
				break;
		}
		gdev_unlock_restore(&vas->lock, &flags);
		break;
	case GDEV_MEM_DMA:
		gdev_lock_save(&vas->lock, &flags);
		gdev_list_for_each (mem, &vas->dma_mem_list, list_entry_heap) {
			uint64_t map_addr = (uint64_t)mem->map;
			if ((addr >= map_addr) && (addr < map_addr + mem->size))
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
void *gdev_mem_getbuf(struct gdev_mem *mem)
{
	return mem->map;
}

/* get virtual memory address. */
uint64_t gdev_mem_getaddr(struct gdev_mem *mem)
{
	return mem->addr;
}

/* get allocated memory size. */
uint64_t gdev_mem_getsize(struct gdev_mem *mem)
{
	return mem->size;
}

/* get physical bus address. */
uint64_t gdev_mem_phys_getaddr(struct gdev_mem *mem, uint64_t offset)
{
	return gdev_raw_mem_phys_getaddr(mem, offset);
}

