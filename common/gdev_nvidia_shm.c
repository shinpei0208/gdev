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
#include "gdev_sched.h"

#define GDEV_SHM_SEGMENT_COUNT 512 /* hardcoded */
static struct gdev_mem *gdev_shm_owners[GDEV_SHM_SEGMENT_COUNT] = {
	[0 ... GDEV_SHM_SEGMENT_COUNT - 1] = NULL
};

/**
 * these functions must be used when saving/loading data.
 * better implementations wanted!
 */
int gdev_callback_save_to_host(void*, void*, uint64_t, uint64_t);
int gdev_callback_save_to_device(void*, uint64_t, uint64_t, uint64_t);
int gdev_callback_load_from_host(void*, uint64_t, void*, uint64_t);
int gdev_callback_load_from_device(void*, uint64_t, uint64_t, uint64_t);

/* attach device memory and allocate host buffer for swap */
static int __gdev_swap_attach(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *swap_mem;
	void *swap_buf;

	/* host buffer for swap. */
	swap_buf = MALLOC(mem->size);
	if (!swap_buf)
		goto fail_swap_buf;
	/* device memory for temporal swap (shared by others). */
	if (GDEV_SWAP_MEM_SIZE > 0) {
		swap_mem = gdev_raw_mem_share(vas, gdev->swap);
		if (!swap_mem)
			goto fail_swap_mem;
		gdev_nvidia_mem_setup(swap_mem, vas, GDEV_MEM_DEVICE);
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

/* detach device memory and free host buffer for swap */
static void __gdev_swap_detach(struct gdev_mem *mem)
{
	if (GDEV_SWAP_MEM_SIZE > 0)
		gdev_raw_mem_unshare(mem->swap_mem);
	FREE(mem->swap_buf);
}

static void __gdev_shm_init(struct gdev_mem *mem, struct gdev_shm *shm)
{
	gdev_list_init(&shm->mem_list, NULL);
	gdev_list_init(&shm->list_entry, (void*)shm);
	gdev_mutex_init(&shm->mutex);
	shm->prio = GDEV_PRIO_MIN;
	shm->users = 1; /* count itself. */
	shm->holder = NULL;
	shm->bo = mem->bo;
	shm->size = mem->size;
	shm->key = 0;
	shm->id = -1;
	shm->implicit = 0;
	gdev_mutex_lock(&shm->mutex);
	/* the victim object itself must be inserted into the list. */
	gdev_list_add(&mem->list_entry_shm, &shm->mem_list);
	mem->shm = shm;
	gdev_mutex_unlock(&shm->mutex);
}

int gdev_shm_create(struct gdev_device *gdev, struct gdev_vas *vas, int key, uint64_t size, int flags)
{
	struct gdev_mem *mem;
	struct gdev_shm *shm;
	int id = -1;
	int i;

	gdev_list_for_each(shm, &gdev->shm_list, list_entry) {
		if (!shm->implicit && key == shm->key) {
			id = shm->id;
			break;
		}
	}

	if (id < 0) {
		/* @size could be resized (aligned by page size) */
		if (!(mem = gdev_raw_mem_alloc(vas, size))) {
			GDEV_PRINT("Failed to allocate memory\n");
			goto fail_mem_alloc;
		}
		gdev_nvidia_mem_setup(mem, vas, GDEV_MEM_DEVICE);
		/* register the new shared memory segment. */
		for (i = 0; i < GDEV_SHM_SEGMENT_COUNT; i++) {
			if (!gdev_shm_owners[i]) {
				gdev_shm_owners[i] = mem;
				id = i;
				if (!(shm = MALLOC(sizeof(*shm)))) {
					GDEV_PRINT("Failed to allocate heap\n");
					goto fail_shm_malloc;
				}
				__gdev_shm_init(mem, shm);
				gdev_list_add(&shm->list_entry, &gdev->shm_list);
				shm->key = key;
				shm->id = id;
				if (id < 0) {
					GDEV_PRINT("Allocated new shared memory segment %d\n", id);
				}
				break;
			}
		}
	}

	if (id < 0) {
		GDEV_PRINT("No space in shared memory segments\n");
	}

	return id;

fail_shm_malloc:
	gdev_mem_free(mem);
fail_mem_alloc:
	return -1;
}

/**
 * mark to destroy the shared memory segment.
 * FIXME: there is a security issue that anyone can destroy it...
 */
int gdev_shm_destroy_mark(struct gdev_device *gdev, struct gdev_mem *owner)
{
	gdev_mutex_lock(&owner->shm->mutex);
	/* find the new owner (could be NULL). */
	gdev_shm_owners[owner->shm->id] = 
		gdev_list_container(gdev_list_head(&owner->shm->mem_list));
	gdev_mutex_unlock(&owner->shm->mutex);
	
	gdev_shm_detach(owner);

	return 0;
}

/* find a memory object that we can borrow some memory space from. */
static struct gdev_mem *__gdev_shm_find_victim(struct gdev_vas *vas, uint64_t size)
{
	struct gdev_device *gdev = vas->gdev;
	struct gdev_vas *v;
	struct gdev_mem *m, *victim = NULL;
	struct gdev_shm *shm;
	unsigned long flags;

	/* select the lowest-priority object. */
	gdev_lock_save(&gdev->vas_lock, &flags);
	gdev_list_for_each (v, &gdev->vas_list, list_entry) {
		gdev_lock_nested(&v->lock);
		gdev_list_for_each (m, &v->mem_list, list_entry_heap) {
			/* don't select from the save VAS object! */
			if (m->size >= size && m->vas != vas) {
				if (!victim)
					victim = m;
				else {
					int victim_prio, m_prio;
					if (victim->shm)
						victim_prio = victim->shm->prio;
					else
						victim_prio = victim->vas->prio;
					if (m->shm)
						m_prio = m->shm->prio;
					else
						m_prio = m->vas->prio;
					if (victim_prio > m_prio)
						victim = m;
				}
			}
		}
		gdev_unlock_nested(&v->lock);
	}
	gdev_unlock_restore(&gdev->vas_lock, &flags);

	/* if the victim object doesn't have a shared memory object yet, 
	   allocate a new one here. note that this shared memory doesn't need
	   a key search, and hence is not inserted into the shared memory list. */
	if (victim && !victim->shm) {
		/* allocate new shared memory object. */
		if (!(shm = MALLOC(sizeof(*shm))))
			return NULL;
		/* attach swap for evicting data from shared memory space. */
		if (__gdev_swap_attach(victim)) {
			FREE(shm);
			return NULL;
		}
		__gdev_shm_init(victim, shm);
		gdev_list_add(&shm->list_entry, &gdev->shm_list);
		shm->implicit = 1; /* shared memory is created implicitly */
	}
	else if (victim) {
		shm = victim->shm;
		if (!shm->implicit) {
			gdev_mutex_lock(&shm->mutex);
			gdev_list_for_each(m, &shm->mem_list, list_entry_shm) {
				if (__gdev_swap_attach(m)) {
					/* if someone fails, detach all. */
					gdev_list_for_each(m, &shm->mem_list, list_entry_shm) {
						if (m->swap_mem)
							__gdev_swap_detach(m);
					}
					gdev_mutex_unlock(&shm->mutex);
					return NULL;
				}
			}
			/* now turns into an implicit shared memory object. */
			shm->implicit = 1; 
			gdev_mutex_unlock(&shm->mutex);
		}
	}

	return victim;
}

/* share memory space with @mem. if @mem is null, find victim instead. 
   the function is protected by gdev->shm_mutex. */
struct gdev_mem *gdev_shm_attach(struct gdev_vas *vas, struct gdev_mem *mem, uint64_t size)
{
	struct gdev_mem *new;
	int implicit;

	if (!mem) {
		/* select a victim memory object. victim->shm will be newly 
		   allocated if NULL. */
		if (!(mem = __gdev_shm_find_victim(vas, size)))
			goto fail_victim;
		implicit = 1;
	}
	else {
		if (!mem->shm) {
			struct gdev_shm *shm;
			if (!(shm = MALLOC(sizeof(*shm))))
				goto fail_heap;
			/* initialize shared memory, but don't add it to the list. */
			__gdev_shm_init(mem, shm);
		}
		implicit = 0;
	}

	/* borrow the same (physical) memory space by sharing. */
	if (!(new = gdev_raw_mem_share(vas, mem)))
		goto fail_shm;

	/* initialize the new memory object. type = mem->type. */
	gdev_nvidia_mem_setup(new, vas, mem->type);

	/* if created implicitly, the object will need eviction at runtime. */
	if (implicit) {
		/* attach swap for evicting data from shared memory space. */
		if (__gdev_swap_attach(new))
			goto fail_swap;
	}

	gdev_mutex_lock(&mem->shm->mutex);
	new->shm = mem->shm;
	/* set the highest prio among users to the shared memory's priority. */
	if (new->shm->prio < vas->prio)
		new->shm->prio = vas->prio;
	/* insert the new memory object into the shared memory list. */
	gdev_list_add(&new->list_entry_shm, &new->shm->mem_list);
	gdev_mutex_unlock(&mem->shm->mutex);

	/* this increment is protected by gdev->shm_mutex. */
	mem->shm->users++;

	gdev_nvidia_mem_list_add(new);

	return new;

fail_swap:
	gdev_raw_mem_unshare(new);
fail_shm:
fail_heap:
fail_victim:

	return NULL;
}

/* detach @mem from the shared memory. 
   this function is protected by gdev->shm_mutex. */
void gdev_shm_detach(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_shm *shm = mem->shm;
	struct gdev_mem *m;

	gdev_nvidia_mem_list_del(mem);

	/* this decrement is protected by gdev->shm_mutex. */
	shm->users--;

	/* enter the local shared memory lock. */
	gdev_mutex_lock(&shm->mutex);
	mem->shm = NULL;
	gdev_list_del(&mem->list_entry_shm);
	if (shm->implicit)
		__gdev_swap_detach(mem);

	/* if the memory object is shared but no users, free it. 
	   since users == 0, no one else will use mem->shm. */
	if (shm->users == 0) {
		/* freeing memory must be exclusive with using shared memory. */
		gdev_raw_mem_free(mem); 
		gdev_mutex_unlock(&shm->mutex);
		gdev_list_del(&shm->list_entry); /* remove from the device shm list. */
		FREE(shm);
	}
	/* otherwise, just unshare the memory object. */
	else {
		/* if a freeing memory object has the highest priority among the 
		   shared memory objects, find the next highest priority one. */
		if (shm->prio == vas->prio) {
			int prio = GDEV_PRIO_MIN;
			gdev_list_for_each (m, &shm->mem_list, list_entry_shm) {
				if (m->vas->prio > prio)
					prio = m->vas->prio;
			}
			shm->prio = prio;
		}
		if (shm->holder == mem)
			shm->holder = NULL;
		/* unsharing memory must be exclusive with using shared memory. */
		gdev_raw_mem_unshare(mem);
		gdev_mutex_unlock(&shm->mutex);
	}
}

/* lookup the owner of the shared memory associated with @id. */
struct gdev_mem *gdev_shm_lookup(struct gdev_device *gdev, int id)
{
	if (id < 0 || id >= GDEV_SHM_SEGMENT_COUNT)
		return NULL;
	return gdev_shm_owners[id];
}

/* evict the conflicting shared memory object data.
   the shared memory object associated with @mem must be locked. */
int gdev_shm_evict_conflict(struct gdev_ctx *ctx, struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_shm *shm = mem->shm;
	struct gdev_device *gdev = vas->gdev;
	struct gdev_mem *holder;
	uint64_t src_addr;
	uint64_t size;
	int ret;

	if (shm && shm->implicit) {
		void *h = ctx->vas->handle;
		if (shm->holder && shm->holder != mem) {
			struct gdev_mem *dev_swap = gdev->swap;
			holder = shm->holder;
			src_addr = mem->addr;
			size = shm->size;
			if (dev_swap && !dev_swap->shm->holder) {
				uint64_t dst_addr = holder->swap_mem->addr;
				ret = gdev_callback_save_to_device(h, dst_addr, src_addr, size);
				if (ret)
					goto fail_evict;
				dev_swap->shm->holder = mem;
			}
			else {
				void *dst_buf = holder->swap_buf;
				ret = gdev_callback_save_to_host(h, dst_buf, src_addr, size);
				if (ret)
					goto fail_evict;
			}
			holder->evicted = 1;
		}
		shm->holder = mem;
	}

	return 0;
	
fail_evict:
	return ret;
}

/* retrieve data evicted in swap space.
   the shared memory object associated with @mem must be locked. */
int gdev_shm_retrieve_swap(struct gdev_ctx *ctx, struct gdev_mem *mem)
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
		gdev_shm_evict_conflict(ctx, mem);
		/* load data regardless whether eviction succeeded or failed. */
		dev_swap = gdev->swap;
		dst_addr = mem->addr;
		size = mem->size;
		if (dev_swap && dev_swap->shm->holder == mem) {
			uint64_t src_addr = mem->swap_mem->addr;
			ret = gdev_callback_load_from_device(h, dst_addr, src_addr, size);
			if (ret)
				goto fail_retrieve;
		}
		else {
			void *src_buf = mem->swap_buf;
			ret = gdev_callback_load_from_host(h, dst_addr, src_buf, size);
			if (ret)
				goto fail_retrieve;
		}
		mem->evicted = 0;
		mem->shm->holder = mem;
	}

	return 0;

fail_retrieve:
	return ret;
}

/* retrieve all data evicted in swap space associated to @vas.
   all the shared memory objects associated to @vas must be locked. */
int gdev_shm_retrieve_swap_all(struct gdev_ctx *ctx, struct gdev_vas *vas)
{
	struct gdev_mem *mem;

	gdev_list_for_each (mem, &vas->mem_list, list_entry_heap) {
		gdev_shm_retrieve_swap(ctx, mem);
	}

	return 0;
}

/* create swap memory object for the device. */
int gdev_swap_create(struct gdev_device *gdev, uint32_t size)
{
	struct gdev_mem *swap;
	struct gdev_shm *shm;
	int ret = 0;

	swap = gdev_raw_swap_alloc(gdev, size);
	if (swap) {
		shm = MALLOC(sizeof(*shm));
		if (shm)
			shm->holder = NULL;
		else {
			gdev_raw_swap_free(swap);
			swap = NULL;
			ret = -ENOMEM;
		}
		swap->shm = shm;
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
		if (gdev->swap->shm)
			FREE(gdev->swap->shm);
		gdev_raw_swap_free(gdev->swap);
	}
}
