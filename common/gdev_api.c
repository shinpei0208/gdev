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
#include "gdev_device.h"
#include "gdev_sched.h"

#define __max(x, y) (x) > (y) ? (x) : (y)
#define __min(x, y) (x) < (y) ? (x) : (y)

/**
 * Gdev handle struct: not visible to outside.
 */
struct gdev_handle {
	struct gdev_device *gdev; /* gdev handle object. */
	struct gdev_sched_entity *se; /* scheduling entity. */
	gdev_vas_t *vas; /* virtual address space object. */
	gdev_ctx_t *ctx; /* device context object. */
	gdev_mem_t **dma_mem; /* host-side DMA memory object (bounce buffer). */
	uint32_t chunk_size; /* configurable memcpy chunk size. */
	int pipeline_count; /* configurable memcpy pipeline count. */
	int dev_id; /* device ID. */
};

static gdev_mem_t** __malloc_dma(gdev_vas_t *vas, uint64_t size, int p_count)
{
	gdev_mem_t **dma_mem;
	int i;

	dma_mem = MALLOC(sizeof(*dma_mem) * p_count);
	if (!dma_mem)
		return NULL;

	for (i = 0; i < p_count; i++) {
		dma_mem[i] = gdev_mem_alloc(vas, size, GDEV_MEM_DMA);
		if (!dma_mem[i])
			return NULL;
	}

	return dma_mem;
}

static void __free_dma(gdev_mem_t **dma_mem, int p_count)
{
	int i;

	for (i = 0; i < p_count; i++)
		gdev_mem_free(dma_mem[i]);

	FREE(dma_mem);
}

/**
 * a wrapper of memcpy().
 */
static int __f_memcpy(void *dst, const void *src, uint32_t size)
{
	memcpy(dst, src, size);
	return 0;
}

/**
 * a wrapper of copy_from_user() - it wraps memcpy() for user-space.
 */
static int __f_cfu(void *dst, const void *src, uint32_t size)
{
	if (COPY_FROM_USER(dst, src, size))
		return -EFAULT;
	return 0;
}

/**
 * a wrapper of copy_to_user() - it wraps memcpy() for user-space.
 */
static int __f_ctu(void *dst, const void *src, uint32_t size)
{
	if (COPY_TO_USER(dst, src, size))
		return -EFAULT;
	return 0;
}

/**
 * copy host buffer to device memory with pipelining.
 * @host_copy is either memcpy() or copy_from_user().
 */
static int __gmemcpy_to_device_p
(gdev_ctx_t *ctx, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int async, uint32_t ch_size, int p_count, gdev_mem_t **bmem,
 int (*host_copy)(void*, const void*, uint32_t))
{
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t dma_size;
	int ret = 0;
	int i;

	for (i = 0; i < p_count; i++) {
		dma_addr[i] = gdev_mem_get_addr(bmem[i]);
		dma_buf[i] = gdev_mem_get_buf(bmem[i]);
		fence[i] = 0;
	}

	offset = 0;
	for (;;) {
		for (i = 0; i < p_count; i++) {
			dma_size = __min(rest_size, ch_size);
			rest_size -= dma_size;
			/* HtoH */
			if (fence[i])
				gdev_poll(ctx, fence[i], NULL);
			ret = host_copy(dma_buf[i], src_buf + offset, dma_size);
			if (ret)
				goto end;
			/* HtoD */
			fence[i] = gdev_memcpy(ctx, dst_addr + offset, dma_addr[i], 
								   dma_size, async);
			if (rest_size == 0) {
				/* wait for the last fence, and go out! */
				gdev_poll(ctx, fence[i], NULL);
				goto end;
			}
			offset += dma_size;
		}
	}

end:
	return ret;
}

/**
 * copy host buffer to device memory without pipelining.
 * @host_copy is either memcpy() or copy_from_user().
 */
static int __gmemcpy_to_device_np
(gdev_ctx_t *ctx, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int async, uint32_t ch_size, gdev_mem_t **bmem, 
 int (*host_copy)(void*, const void*, uint32_t))
{
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

	dma_addr[0] = gdev_mem_get_addr(bmem[0]);
	dma_buf[0] = gdev_mem_get_buf(bmem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, ch_size);
		ret = host_copy(dma_buf[0], src_buf + offset, dma_size);
		if (ret)
			goto end;
		fence = gdev_memcpy(ctx, dst_addr + offset, dma_addr[0], 
							dma_size, async);
		gdev_poll(ctx, fence, NULL);
		rest_size -= dma_size;
		offset += dma_size;
	}

end:
	return ret;
}

/**
 * copy host DMA buffer to device memory.
 */
static int __gmemcpy_dma_to_device
(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint64_t size, 
 int async)
{
	uint32_t fence;

	/* we don't break data into chunks if copying directly from dma memory. */
	fence = gdev_memcpy(ctx, dst_addr, src_addr, size, async);
	gdev_poll(ctx, fence, NULL);
	
	return 0;
}

/**
 * a wrapper function of __gmemcpy_to_device().
 */
static int __gmemcpy_to_device_locked
(gdev_ctx_t *ctx, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int async, uint32_t ch_size, int p_count, gdev_vas_t *vas, gdev_mem_t *mem, 
 gdev_mem_t **dma_mem, int (*host_copy)(void*, const void*, uint32_t))
{
	gdev_mem_t *hmem;
	gdev_mem_t **bmem;
	int ret;

	if (size <= 4) {
		gdev_write32(mem, dst_addr, ((uint32_t*)src_buf)[0]);
		ret = 0;
	}
	else if (size <= GDEV_MEMCPY_IORW_LIMIT) {
		ret = gdev_write(mem, dst_addr, src_buf, size);
	}
	else if ((hmem = gdev_mem_lookup(vas, (uint64_t)src_buf, GDEV_MEM_DMA))) {
		ret = __gmemcpy_dma_to_device(ctx, dst_addr, hmem->addr, size, async);
	}
	else {
		/* prepare bounce buffer memory. */
		if (!dma_mem) {
			bmem = __malloc_dma(vas, __min(size, ch_size), p_count);
			if (!bmem)
				return -ENOMEM;
		}
		else
			bmem = dma_mem;

		/* copy memory to device. */
		if (p_count > 1 && size > ch_size)
			ret = __gmemcpy_to_device_p(ctx, dst_addr, src_buf, size, async,
										ch_size, p_count, bmem, host_copy);
		else
			ret = __gmemcpy_to_device_np(ctx, dst_addr, src_buf, size, async,
										 ch_size, bmem, host_copy);

		/* free bounce buffer memory, if necessary. */
		if (!dma_mem)
			__free_dma(bmem, p_count);
	}

	return ret;
}

/**
 * a wrapper function of gmemcpy_to_device().
 */
static int __gmemcpy_to_device
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size,
 int async, int (*host_copy)(void*, const void*, uint32_t))
{
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;
	gdev_mem_t *mem = gdev_mem_lookup(vas, dst_addr, GDEV_MEM_DEVICE);
	gdev_mem_t **dma_mem = h->dma_mem;
	uint32_t ch_size = h->chunk_size;
	int p_count = h->pipeline_count;
	int ret;

	if (!mem)
		return -ENOENT;

	gdev_mem_lock(mem);
	gdev_shm_evict_conflict(ctx, mem); /* evict conflicting data. */
	ret = __gmemcpy_to_device_locked(ctx, dst_addr, src_buf, size, async, 
									 ch_size, p_count, vas, mem, dma_mem, 
									 host_copy);
	gdev_mem_unlock(mem);
	
	return ret;
}

/**
 * copy device memory to host buffer with pipelining.
 * host_copy() is either memcpy() or copy_to_user().
 */
static int __gmemcpy_from_device_p
(gdev_ctx_t *ctx, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int async, uint32_t ch_size, int p_count, gdev_mem_t **bmem,
 int (*host_copy)(void*, const void*, uint32_t))
{
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t dma_size;
	int ret = 0;
	int i;

	for (i = 0; i < p_count; i++) {
		dma_addr[i] = gdev_mem_get_addr(bmem[i]);
		dma_buf[i] = gdev_mem_get_buf(bmem[i]);
		fence[i] = 0;
	}

	offset = 0;
	dma_size = __min(rest_size, ch_size);
	rest_size -= dma_size;
	/* DtoH */
	fence[0] = gdev_memcpy(ctx, dma_addr[0], src_addr + 0, dma_size, async);
	for (;;) {
		for (i = 0; i < p_count; i++) {
			if (rest_size == 0) {
				/* HtoH */
				gdev_poll(ctx, fence[i], NULL);
				host_copy(dst_buf + offset, dma_buf[i], dma_size);
				goto end;
			}
			dma_size = __min(rest_size, ch_size);
			rest_size -= dma_size;
			offset += dma_size;
			/* DtoH */
			if (i + 1 == p_count)
				fence[0] = gdev_memcpy(ctx, dma_addr[0], src_addr + offset,
									   dma_size, async);
			else
				fence[i+1] = gdev_memcpy(ctx, dma_addr[i+1], src_addr + offset,
										 dma_size, async);
			/* HtoH */
			gdev_poll(ctx, fence[i], NULL);
			ret = host_copy(dst_buf + offset - dma_size, dma_buf[i], dma_size);
			if (ret)
				goto end;
		}
	}

end:
	return ret;
}

/**
 * copy device memory to host buffer without pipelining.
 * host_copy() is either memcpy() or copy_to_user().
 */
static int __gmemcpy_from_device_np
(gdev_ctx_t *ctx, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int async, uint32_t ch_size, gdev_mem_t **bmem,
 int (*host_copy)(void*, const void*, uint32_t))
{
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

	dma_addr[0] = gdev_mem_get_addr(bmem[0]);
	dma_buf[0] = gdev_mem_get_buf(bmem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, ch_size);
		fence = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, 
							dma_size, async);
		gdev_poll(ctx, fence, NULL);
		ret = host_copy(dst_buf + offset, dma_buf[0], dma_size);
		if (ret)
			goto end;
		rest_size -= dma_size;
		offset += dma_size;
	}

end:
	return ret;
}

/**
 * copy device memory to host DMA buffer.
 */
static int __gmemcpy_dma_from_device
(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint64_t size,
 int async)
{
	uint32_t fence;

	/* we don't break data into chunks if copying directly from dma memory. */
	fence = gdev_memcpy(ctx, dst_addr, src_addr, size, async); 
	gdev_poll(ctx, fence, NULL);

	return 0;
}

/**
 * a wrapper function of __gmemcpy_from_device().
 */
static int __gmemcpy_from_device_locked
(gdev_ctx_t *ctx, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int async, uint32_t ch_size, int p_count, gdev_vas_t *vas, gdev_mem_t *mem, 
 gdev_mem_t **dma_mem, int (*host_copy)(void*, const void*, uint32_t))
{
	gdev_mem_t *hmem;
	gdev_mem_t **bmem;
	int ret;

	if (size <= 4) {
		((uint32_t*)dst_buf)[0] = gdev_read32(mem, src_addr);
		ret = 0;
	}
	else if (size <= GDEV_MEMCPY_IORW_LIMIT) {
		ret = gdev_read(mem, dst_buf, src_addr, size);
	}
	else if ((hmem = gdev_mem_lookup(vas, (uint64_t)dst_buf, GDEV_MEM_DMA))) {
		ret = __gmemcpy_dma_from_device(ctx, hmem->addr, src_addr, size, async);
	}
	else {
		/* prepare bounce buffer memory. */
		if (!dma_mem) {
			bmem = __malloc_dma(vas, __min(size, ch_size), p_count);
			if (!bmem)
				return -ENOMEM;
		}
		else
			bmem = dma_mem;

		if (p_count > 1 && size > ch_size)
			ret = __gmemcpy_from_device_p(ctx, dst_buf, src_addr, size, async,
										  ch_size, p_count, bmem, host_copy);
		else
			ret = __gmemcpy_from_device_np(ctx, dst_buf, src_addr, size, async,
										   ch_size, bmem, host_copy);

		/* free bounce buffer memory, if necessary. */
		if (!dma_mem)
			__free_dma(bmem, p_count);
	}

	return ret;
}

/**
 * a wrapper function of gmemcpy_from_device().
 */
static int __gmemcpy_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int async, int (*host_copy)(void*, const void*, uint32_t))
{
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;
	gdev_mem_t *mem = gdev_mem_lookup(vas, src_addr, GDEV_MEM_DEVICE);
	gdev_mem_t **dma_mem = h->dma_mem;
	uint32_t ch_size = h->chunk_size;
	int p_count = h->pipeline_count;
	int ret;

	if (!mem)
		return -ENOENT;

	gdev_mem_lock(mem);
	gdev_shm_retrieve_swap(ctx, mem); /* retrieve data swapped. */
	ret = __gmemcpy_from_device_locked(ctx, dst_buf, src_addr, size, async, 
									   ch_size, p_count, vas, mem, dma_mem,
									   host_copy);
	gdev_mem_unlock(mem);

	return ret;
}

/**
 * this function must be used when saving data to host.
 */
int gdev_callback_save_to_host
(void *h, void* dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_vas_t *vas = ((struct gdev_handle*)h)->vas;
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	gdev_mem_t *mem = gdev_mem_lookup(vas, src_addr, GDEV_MEM_DEVICE);
	gdev_mem_t **dma_mem = ((struct gdev_handle*)h)->dma_mem;
	uint32_t ch_size = ((struct gdev_handle*)h)->chunk_size;
	int p_count = ((struct gdev_handle*)h)->pipeline_count;

	return __gmemcpy_from_device_locked(ctx, dst_buf, src_addr, size, 1, 
										ch_size, p_count, vas, mem, dma_mem,
										__f_memcpy);
}

/**
 * this function must be used when saving data to device.
 */
int gdev_callback_save_to_device
(void *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	uint32_t fence;

	fence = gdev_memcpy(ctx, dst_addr, src_addr, size, 0);
	gdev_poll(ctx, fence, NULL);

	return 0;
}

/**
 * this function must be used when loading data from host.
 */
int gdev_callback_load_from_host
(void *h, uint64_t dst_addr, void *src_buf, uint64_t size)
{
	gdev_vas_t *vas = ((struct gdev_handle*)h)->vas;
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	gdev_mem_t *mem = gdev_mem_lookup(vas, dst_addr, GDEV_MEM_DEVICE);
	gdev_mem_t **dma_mem = ((struct gdev_handle*)h)->dma_mem;
	uint32_t ch_size = ((struct gdev_handle*)h)->chunk_size;
	int p_count = ((struct gdev_handle*)h)->pipeline_count;

	return __gmemcpy_to_device_locked(ctx, dst_addr, src_buf, size, 1,
									  ch_size, p_count, vas, mem, dma_mem,
									  __f_memcpy);
}

/**
 * this function must be used when loading data from device.
 */
int gdev_callback_load_from_device
(void *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	uint32_t fence;

	fence = gdev_memcpy(ctx, dst_addr, src_addr, size, 0);
	gdev_poll(ctx, fence, NULL);

	return 0;
}

/******************************************************************************
 ******************************************************************************
 * Gdev API functions
 ******************************************************************************
 ******************************************************************************/

/**
 * gopen():
 * create a new GPU context on the given device #@devnum.
 */
struct gdev_handle *gopen(int minor)
{
	struct gdev_handle *h;
	struct gdev_device *gdev;
	struct gdev_sched_entity *se;
	gdev_vas_t *vas;
	gdev_ctx_t *ctx;
	gdev_mem_t **dma_mem;

	if (!(h = MALLOC(sizeof(*h))))
		return NULL;

	h->pipeline_count = GDEV_PIPELINE_DEFAULT_COUNT;
	h->chunk_size = GDEV_CHUNK_DEFAULT_SIZE;

	/* open the specified device. */
	gdev = gdev_dev_open(minor);
	if (!gdev) {
		GDEV_PRINT("Failed to open gdev%d\n", minor);
		goto fail_open;
	}

	/* create a new virual address space (VAS) object. */
	vas = gdev_vas_new(gdev, GDEV_VAS_SIZE, h);
	if (!vas) {
		GDEV_PRINT("Failed to create a virtual address space object\n");
		goto fail_vas;
	}

	/* create a new GPU context object. */
	ctx = gdev_ctx_new(gdev, vas);
	if (!ctx) {
		GDEV_PRINT("Failed to create a context object\n");
		goto fail_ctx;
	}

	/* allocate static bounce bound buffer objects. */
	dma_mem = __malloc_dma(vas, GDEV_CHUNK_DEFAULT_SIZE, h->pipeline_count);
	if (!dma_mem) {
		GDEV_PRINT("Failed to allocate static DMA buffer object\n");
		goto fail_dma;
	}

	/* allocate a scheduling entity. */
	se = gdev_sched_entity_create(gdev, ctx);
	if (!se) {
		GDEV_PRINT("Failed to allocate scheduling entity\n");
		goto fail_se;
	}
	
	/* save the objects to the handle. */
	h->se = se;
	h->dma_mem = dma_mem;
	h->vas = vas;
	h->ctx = ctx;
	h->gdev = gdev;
	h->dev_id = minor;

	GDEV_PRINT("Opened gdev%d\n", minor);

	return h;

fail_se:
	__free_dma(dma_mem, h->pipeline_count);
fail_dma:
	gdev_ctx_free(ctx);
fail_ctx:
	gdev_vas_free(vas);
fail_vas:
	gdev_dev_close(gdev);
fail_open:
	return NULL;
}

/**
 * gclose():
 * destroy the GPU context associated with @handle.
 */
int gclose(struct gdev_handle *h)
{
	if (!h)
		return -ENOENT;
	if (!h->gdev || !h->ctx || !h->vas || !h->se)
		return -ENOENT;

	/* free the scheduling entity. */
	gdev_sched_entity_destroy(h->se);
	
	/* free the bounce buffer. */
	if (h->dma_mem)
		__free_dma(h->dma_mem, h->pipeline_count);

	/* garbage collection: free all memory left in heap. */
	gdev_mem_gc(h->vas);

	/* free the objects. */
	gdev_ctx_free(h->ctx);
	gdev_vas_free(h->vas);
	gdev_dev_close(h->gdev);

	GDEV_PRINT("Closed gdev%d\n", h->dev_id);

	FREE(h);

	return 0;
}

/**
 * gmalloc():
 * allocate new device memory space.
 */
uint64_t gmalloc(struct gdev_handle *h, uint64_t size)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;

	gdev->mem_used += size;

	if (gdev->mem_used > gdev->mem_size) {
		/* try to share memory with someone (only for device memory). 
		   the shared memory must be freed in gdev_mem_free() when 
		   unreferenced by all users. */
		if (!(mem = gdev_mem_share(vas, size)))
			goto fail;
	}
	else if (!(mem = gdev_mem_alloc(vas, size, GDEV_MEM_DEVICE)))
		goto fail;

	return gdev_mem_get_addr(mem);

fail:
	gdev->mem_used -= size;
	return 0;
}

/**
 * gfree():
 * free the memory space allocated at the specified address.
 */
uint64_t gfree(struct gdev_handle *h, uint64_t addr)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t size;

	if (!(mem = gdev_mem_lookup(vas, addr, GDEV_MEM_DEVICE)))
		goto fail;
	size = gdev_mem_get_size(mem);
	gdev_mem_free(mem);

	gdev->mem_used -= size;

	return size;

fail:
	return 0;
}

/**
 * gmalloc_dma():
 * allocate new host dma memory space.
 */
void *gmalloc_dma(struct gdev_handle *h, uint64_t size)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;

	gdev->dma_mem_used += size;

	if (gdev->dma_mem_used > gdev->dma_mem_size)
		goto fail;
	else if (!(mem = gdev_mem_alloc(vas, size, GDEV_MEM_DMA)))
		goto fail;

	return gdev_mem_get_buf(mem);

fail:
	gdev->dma_mem_used -= size;
	return 0;
}

/**
 * gfree_dma():
 * free the host dma memory space allocated at the specified buffer.
 */
uint64_t gfree_dma(struct gdev_handle *h, void *buf)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t size;

	if (!(mem = gdev_mem_lookup(vas, (uint64_t)buf, GDEV_MEM_DMA)))
		goto fail;
	size = gdev_mem_get_size(mem);
	gdev_mem_free(mem);

	gdev->dma_mem_used -= size;

	return size;

fail:
	return 0;
}

/**
 * gmemcpy_to_device():
 * copy data from @buf to device memory at @addr.
 */
int gmemcpy_to_device
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	/* async = false and host memcpy will use memcpy(). */
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, 0, __f_memcpy);
}

/**
 * gmemcpy_to_device_async():
 * asynchronously copy data from @buf to device memory at @addr.
 */
int gmemcpy_to_device_async
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	/* async = true and host memcpy will use memcpy(). */
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, 1, __f_memcpy);
}

/**
 * gmemcpy_user_to_device():
 * copy data from "user-space" @buf to device memory at @addr.
 */
int gmemcpy_user_to_device
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	/* async = false and host memcpy will use copy_from_user(). */
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, 0, __f_cfu);
}

/**
 * gmemcpy_user_to_device_async():
 * asynchrounouly copy data from "user-space" @buf to device memory at @addr.
 */
int gmemcpy_user_to_device_async
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	/* async = true and host memcpy will use copy_from_user(). */
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, 1, __f_cfu);
}

/**
 * gmemcpy_from_device():
 * copy data from device memory at @addr to @buf.
 */
int gmemcpy_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	/* async = false and host memcpy will use memcpy(). */
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, 0, __f_memcpy);
}

/**
 * gmemcpy_from_device_async():
 * asynchronously copy data from device memory at @addr to @buf.
 */
int gmemcpy_from_device_async
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	/* async = true and host memcpy will use memcpy(). */
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, 0, __f_memcpy);
}

/**
 * gmemcpy_user_from_device():
 * copy data from device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	/* async = false and host memcpy will use copy_to_user(). */
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, 0, __f_ctu);
}

/**
 * gmemcpy_user_from_device_async():
 * asynchronously copy data from device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device_async
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	/* async = true and host memcpy will use copy_to_user(). */
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, 1, __f_ctu);
}

/**
 * gmemcpy_in_device():
 * copy data of the given size within the device memory.
 */
int gmemcpy_in_device
(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = h->ctx;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *dst = gdev_mem_lookup(vas, dst_addr, GDEV_MEM_DEVICE);
	gdev_mem_t *src = gdev_mem_lookup(vas, src_addr, GDEV_MEM_DEVICE);
	uint32_t fence;

	if (!dst || !src)
		return -ENOENT;

	gdev_mem_lock(dst);
	gdev_mem_lock(src);
	fence = gdev_memcpy(ctx, dst_addr, src_addr, size, 0); 
	gdev_poll(ctx, fence, NULL);
	gdev_mem_unlock(src);
	gdev_mem_unlock(dst);

	return 0;
}

/**
 * glaunch():
 * launch the GPU kernel code.
 */
int glaunch(struct gdev_handle *h, struct gdev_kernel *kernel, uint32_t *id)
{
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;
	struct gdev_sched_entity *se = h->se;

	gdev_schedule_launch(se);

	gdev_mem_lock_all(vas);
	gdev_shm_retrieve_swap_all(ctx, vas); /* get all data swapped back! */
	*id = gdev_launch(ctx, kernel);
	gdev_mem_unlock_all(vas);

	return 0;
}

/**
 * gsync():
 * poll until the GPU becomes available.
 * @timeout is a unit of milliseconds.
 */
int gsync(struct gdev_handle *h, uint32_t id, struct gdev_time *timeout)
{
	return gdev_poll(h->ctx, id, timeout);
}

/**
 * gquery():
 * query the device-specific information.
 */
int gquery(struct gdev_handle *h, uint32_t type, uint64_t *result)
{
	return gdev_query(h->gdev, type, result);
}

/**
 * gtune():
 * tune resource management parameters.
 */
int gtune(struct gdev_handle *h, uint32_t type, uint32_t value)
{
	switch (type) {
	case GDEV_TUNE_MEMCPY_PIPELINE_COUNT:
		if (value > GDEV_PIPELINE_MAX_COUNT || value < GDEV_PIPELINE_MIN_COUNT)
			return -EINVAL;

		if (h->dma_mem)
			__free_dma(h->dma_mem, h->pipeline_count);

		/* change the pipeline count here. */
		h->pipeline_count = value;

		/* reallocate host DMA memory. */
		h->dma_mem = __malloc_dma(h->vas, h->chunk_size, h->pipeline_count);
		if (!h->dma_mem)
			return -ENOMEM;

		break;
	case GDEV_TUNE_MEMCPY_CHUNK_SIZE:
		if (value > GDEV_CHUNK_MAX_SIZE)
			return -EINVAL;

		if (h->dma_mem)
			__free_dma(h->dma_mem, h->pipeline_count);

		/* change the chunk size here. */
		h->chunk_size = value;

		/* reallocate host DMA memory. */
		h->dma_mem = __malloc_dma(h->vas, h->chunk_size, h->pipeline_count);
		if (!h->dma_mem)
			return -ENOMEM;

		break;
	default:
		return -EINVAL;
	}
	return 0;
}

int gshmget(Ghandle h, int key, uint64_t size, int flags)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	int id;

	gdev_mutex_lock(&gdev->shm_mutex);
	id = gdev_shm_create(gdev, vas, key, size, flags);
	gdev_mutex_unlock(&gdev->shm_mutex);

	return id;
}

/**
 * gshmat():
 * attach device shared memory.
 * note that @addr and @flags are currently not supported.
 */
uint64_t gshmat(Ghandle h, int id, uint64_t addr, int flags)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *new, *owner;

	gdev_mutex_lock(&gdev->shm_mutex);
	if (!(owner = gdev_shm_lookup(gdev, id)))
		goto fail;
	if (!(new = gdev_shm_attach(vas, owner, gdev_mem_get_size(owner))))
		goto fail;
	gdev_mutex_unlock(&gdev->shm_mutex);

	return gdev_mem_get_addr(new);

fail:
	gdev_mutex_unlock(&gdev->shm_mutex);
	return 0;
}

/**
 * gshmdt():
 * detach device shared memory.
 */
int gshmdt(Ghandle h, uint64_t addr)
{
	struct gdev_device *gdev = h->gdev;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;

	gdev_mutex_lock(&gdev->shm_mutex);
	if (!(mem = gdev_mem_lookup(vas, addr, GDEV_MEM_DEVICE)))
		goto fail;
	gdev_shm_detach(mem);
	gdev_mutex_unlock(&gdev->shm_mutex);

	return 0;

fail:
	gdev_mutex_unlock(&gdev->shm_mutex);
	return -ENOENT;
}

/**
 * gshmctl():
 * control device shared memory.
 */
int gshmctl(Ghandle h, int id, int cmd, void *buf)
{
	struct gdev_device *gdev = h->gdev;
	gdev_mem_t *owner;
	int ret;
	
	switch (cmd) {
	case GDEV_IPC_RMID:
		gdev_mutex_lock(&gdev->shm_mutex);
		if (!(owner = gdev_shm_lookup(gdev, id))) {
			ret = -ENOENT;
			goto fail;
		}
		gdev_shm_destroy_mark(gdev, owner);
		gdev_mutex_unlock(&gdev->shm_mutex);
		break;
	default:
		GDEV_PRINT("gshmctl(): cmd %d not supported\n", cmd);
		return -EINVAL;
	}

	return 0;

fail:
	gdev_mutex_unlock(&gdev->shm_mutex);
	return ret;
}
