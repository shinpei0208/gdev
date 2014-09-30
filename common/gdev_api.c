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
		if (!dma_mem[i]) {
			while(--i >= 0)
				gdev_mem_free(dma_mem[i]);
			FREE(dma_mem);
			return NULL;
		}
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
static int __gmemcpy_to_device_p(gdev_ctx_t *ctx, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t ch_size, int p_count, gdev_mem_t **bmem, int (*host_copy)(void*, const void*, uint32_t))
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
		dma_addr[i] = gdev_mem_getaddr(bmem[i]);
		dma_buf[i] = gdev_mem_getbuf(bmem[i]);
		fence[i] = 0;
	}

	offset = 0;
	for (;;) {
		for (i = 0; i < p_count; i++) {
			dma_size = __min(rest_size, ch_size);
			/* HtoH */
			if (fence[i])
				gdev_poll(ctx, fence[i], NULL);
			ret = host_copy(dma_buf[i], src_buf+offset, dma_size);
			if (ret)
				goto end;
			/* HtoD */
			fence[i] = gdev_memcpy(ctx, dst_addr+offset, dma_addr[i], dma_size);
			if (rest_size == dma_size) {
				/* wait for the last fence, and go out! */
				gdev_poll(ctx, fence[i], NULL);
				goto end;
			}
			offset += dma_size;
			rest_size -= dma_size;
		}
	}

end:
	return ret;
}

/**
 * copy host buffer to device memory without pipelining.
 * @host_copy is either memcpy() or copy_from_user().
 */
static int __gmemcpy_to_device_np(gdev_ctx_t *ctx, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t ch_size, gdev_mem_t **bmem, int (*host_copy)(void*, const void*, uint32_t))
{
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

	dma_addr[0] = gdev_mem_getaddr(bmem[0]);
	dma_buf[0] = gdev_mem_getbuf(bmem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, ch_size);
		ret = host_copy(dma_buf[0], src_buf + offset, dma_size);
		if (ret)
			goto end;
		fence = gdev_memcpy(ctx, dst_addr + offset, dma_addr[0], dma_size);
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
static int __gmemcpy_dma_to_device(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint64_t size, uint32_t *id)
{
	uint32_t fence;

	/* we don't break data into chunks if copying directly from dma memory. 
	   if @id == NULL, it means memcpy is synchronous. */
	if (!id) {
		fence = gdev_memcpy(ctx, dst_addr, src_addr, size);
		gdev_poll(ctx, fence, NULL);
	}
	else {
		fence = gdev_memcpy_async(ctx, dst_addr, src_addr, size);
		*id = fence;
	}
	
	return 0;
}

/**
 * a wrapper function of __gmemcpy_to_device().
 */
static int __gmemcpy_to_device_locked(gdev_ctx_t *ctx, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id, uint32_t ch_size, int p_count, gdev_vas_t *vas, gdev_mem_t *mem, gdev_mem_t **dma_mem, int (*host_copy)(void*, const void*, uint32_t))
{
	gdev_mem_t *hmem;
	gdev_mem_t **bmem;
	int ret;

	if (size <= 4 && mem->map) {
		gdev_write32(mem, dst_addr, ((uint32_t*)src_buf)[0]);
		ret = 0;
		/* if @id is give while not asynchronous, give it zero. */
		if (id)
			*id = 0;
	}
	else if (size <= GDEV_MEMCPY_IOWRITE_LIMIT && mem->map) {
		ret = gdev_write(mem, dst_addr, src_buf, size);
		/* if @id is give while not asynchronous, give it zero. */
		if (id)
			*id = 0;
	}
	else if ((hmem = gdev_mem_lookup_by_buf(vas, src_buf, GDEV_MEM_DMA))) {
		ret = __gmemcpy_dma_to_device(ctx, dst_addr, hmem->addr, size, id);
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
			ret = __gmemcpy_to_device_p(ctx, dst_addr, src_buf, size, ch_size, p_count, bmem, host_copy);
		else
			ret = __gmemcpy_to_device_np(ctx, dst_addr, src_buf, size, ch_size, bmem, host_copy);

		/* free bounce buffer memory, if necessary. */
		if (!dma_mem)
			__free_dma(bmem, p_count);

		/* if @id is give while not asynchronous, give it zero. */
		if (id)
			*id = 0;
	}

	return ret;
}

/**
 * a wrapper function of gmemcpy_to_device().
 */
static int __gmemcpy_to_device(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id, int (*host_copy)(void*, const void*, uint32_t))
{
#ifndef GDEV_SCHED_DISABLED
	struct gdev_sched_entity *se = h->se;
	struct gdev_device *gdev = h->gdev;
#endif
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;
	gdev_mem_t **dma_mem = h->dma_mem;
	gdev_mem_t *mem;
	uint32_t ch_size = h->chunk_size;
	int p_count = h->pipeline_count;
	int ret;

	mem = gdev_mem_lookup_by_addr(vas, dst_addr, GDEV_MEM_DEVICE);
	if (!mem)
		return -ENOENT;

#ifndef GDEV_SCHED_DISABLED
	/* decide if the context needs to stall or not. */
	gdev_schedule_memory(se);
#endif

	gdev_mem_lock(mem);

	gdev_shm_evict_conflict(ctx, mem); /* evict conflicting data. */
	ret = __gmemcpy_to_device_locked(ctx, dst_addr, src_buf, size, id, ch_size, p_count, vas, mem, dma_mem, host_copy);

	gdev_mem_unlock(mem);

#ifndef GDEV_SCHED_DISABLED
	/* select the next context by itself, since memcpy is sychronous. */
	gdev_select_next_memory(gdev);
#endif

	return ret;
}

/**
 * copy device memory to host buffer with pipelining.
 * host_copy() is either memcpy() or copy_to_user().
 */
static int __gmemcpy_from_device_p(gdev_ctx_t *ctx, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t ch_size, int p_count, gdev_mem_t **bmem, int (*host_copy)(void*, const void*, uint32_t))
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
		dma_addr[i] = gdev_mem_getaddr(bmem[i]);
		dma_buf[i] = gdev_mem_getbuf(bmem[i]);
		fence[i] = 0;
	}

	/* DtoH for all bounce buffers first. */
	offset = 0;
	for (i = 0; i < p_count; i++) {
		dma_size = __min(rest_size, ch_size);
		fence[i] = gdev_memcpy(ctx, dma_addr[i], src_addr + offset, dma_size);
		if (rest_size == dma_size)
			break;
		offset += dma_size;
		rest_size -= dma_size;
	}

	/* reset offset and rest size. */
	offset = 0;
	rest_size = size;
	/* now start overlapping. */
	for (;;) {
		for (i = 0; i < p_count; i++) {
			dma_size = __min(rest_size, ch_size);
			/* HtoH */
			gdev_poll(ctx, fence[i], NULL);
			ret = host_copy(dst_buf + offset, dma_buf[i], dma_size);
			if (ret)
				goto end;
			/* DtoH for the next round if necessary. */
			if (p_count * ch_size < rest_size) {
				uint64_t rest_size_n = rest_size - p_count * ch_size;
				uint32_t dma_size_n = __min(rest_size_n, ch_size);
				uint64_t offset_n = offset + p_count * ch_size;
				fence[i] = gdev_memcpy(ctx, dma_addr[i], src_addr + offset_n, dma_size_n);
			}
			else if (rest_size == dma_size)
				goto end;
			offset += dma_size;
			rest_size -= dma_size;
		}
	}

end:
	return ret;
}

/**
 * copy device memory to host buffer without pipelining.
 * host_copy() is either memcpy() or copy_to_user().
 */
static int __gmemcpy_from_device_np(gdev_ctx_t *ctx, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t ch_size, gdev_mem_t **bmem, int (*host_copy)(void*, const void*, uint32_t))
{
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

	dma_addr[0] = gdev_mem_getaddr(bmem[0]);
	dma_buf[0] = gdev_mem_getbuf(bmem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, ch_size);
		fence = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, dma_size);
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
static int __gmemcpy_dma_from_device(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint64_t size, uint32_t *id)
{
	uint32_t fence;

	/* we don't break data into chunks if copying directly from dma memory. 
	   if @id == NULL, it means memcpy is synchronous. */
	if (!id) {
		fence = gdev_memcpy(ctx, dst_addr, src_addr, size);
		gdev_poll(ctx, fence, NULL);
	}
	else {
		fence = gdev_memcpy_async(ctx, dst_addr, src_addr, size);
		*id = fence;
	}

	return 0;
}

/**
 * a wrapper function of __gmemcpy_from_device().
 */
static int __gmemcpy_from_device_locked(gdev_ctx_t *ctx, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id, uint32_t ch_size, int p_count, gdev_vas_t *vas, gdev_mem_t *mem, gdev_mem_t **dma_mem, int (*host_copy)(void*, const void*, uint32_t))
{
	gdev_mem_t *hmem;
	gdev_mem_t **bmem;
	int ret;

	if (size <= 4 && mem->map) {
		((uint32_t*)dst_buf)[0] = gdev_read32(mem, src_addr);
		ret = 0;
		/* if @id is given despite not asynchronous, give it zero. */
		if (id)
			*id = 0;
	}
	else if (size <= GDEV_MEMCPY_IOREAD_LIMIT && mem->map) {
		ret = gdev_read(mem, dst_buf, src_addr, size);
		/* if @id is given despite not asynchronous, give it zero. */
		if (id)
			*id = 0;
	}
	else if ((hmem = gdev_mem_lookup_by_buf(vas, dst_buf, GDEV_MEM_DMA))) {
		ret = __gmemcpy_dma_from_device(ctx, hmem->addr, src_addr, size, id);
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
			ret = __gmemcpy_from_device_p(ctx, dst_buf, src_addr, size, ch_size, p_count, bmem, host_copy);
		else
			ret = __gmemcpy_from_device_np(ctx, dst_buf, src_addr, size, ch_size, bmem, host_copy);

		/* free bounce buffer memory, if necessary. */
		if (!dma_mem)
			__free_dma(bmem, p_count);

		/* if @id is give while not asynchronous, give it zero. */
		if (id)
			*id = 0;
	}

	return ret;
}

/**
 * a wrapper function of gmemcpy_from_device().
 */
static int __gmemcpy_from_device(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id, int (*host_copy)(void*, const void*, uint32_t))
{
#ifndef GDEV_SCHED_DISABLED
	struct gdev_sched_entity *se = h->se;
	struct gdev_device *gdev = h->gdev;
#endif
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;
	gdev_mem_t **dma_mem = h->dma_mem;
	gdev_mem_t *mem;
	uint32_t ch_size = h->chunk_size;
	int p_count = h->pipeline_count;
	int ret;

	mem = gdev_mem_lookup_by_addr(vas, src_addr, GDEV_MEM_DEVICE);
	if (!mem)
		return -ENOENT;

#ifndef GDEV_SCHED_DISABLED
	/* decide if the context needs to stall or not. */
	gdev_schedule_memory(se);
#endif

	gdev_mem_lock(mem);

	gdev_shm_retrieve_swap(ctx, mem); /* retrieve data swapped. */
	ret = __gmemcpy_from_device_locked(ctx, dst_buf, src_addr, size, id, 
									   ch_size, p_count, vas, mem, dma_mem,
									   host_copy);
	gdev_mem_unlock(mem);

#ifndef GDEV_SCHED_DISABLED
	/* select the next context by itself, since memcpy is synchronous. */
	gdev_select_next_memory(gdev);
#endif

	return ret;
}

/**
 * this function must be used when saving data to host.
 */
int gdev_callback_save_to_host(void *h, void* dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_vas_t *vas = ((struct gdev_handle*)h)->vas;
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	gdev_mem_t **dma_mem = ((struct gdev_handle*)h)->dma_mem;
	gdev_mem_t *mem;
	uint32_t ch_size = ((struct gdev_handle*)h)->chunk_size;
	int p_count = ((struct gdev_handle*)h)->pipeline_count;

	mem = gdev_mem_lookup_by_addr(vas, src_addr, GDEV_MEM_DEVICE);
	if (!mem)
		return -ENOENT;

	return __gmemcpy_from_device_locked(ctx, dst_buf, src_addr, size, NULL, ch_size, p_count, vas, mem, dma_mem, __f_memcpy);
}

/**
 * this function must be used when saving data to device.
 */
int gdev_callback_save_to_device(void *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	uint32_t fence;

	fence = gdev_memcpy(ctx, dst_addr, src_addr, size);
	gdev_poll(ctx, fence, NULL);

	return 0;
}

/**
 * this function must be used when loading data from host.
 */
int gdev_callback_load_from_host(void *h, uint64_t dst_addr, void *src_buf, uint64_t size)
{
	gdev_vas_t *vas = ((struct gdev_handle*)h)->vas;
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	gdev_mem_t **dma_mem = ((struct gdev_handle*)h)->dma_mem;
	gdev_mem_t *mem;
	uint32_t ch_size = ((struct gdev_handle*)h)->chunk_size;
	int p_count = ((struct gdev_handle*)h)->pipeline_count;

	mem = gdev_mem_lookup_by_addr(vas, dst_addr, GDEV_MEM_DEVICE);
	if (!mem)
		return -ENOENT;

	return __gmemcpy_to_device_locked(ctx, dst_addr, src_buf, size, NULL, ch_size, p_count, vas, mem, dma_mem, __f_memcpy);
}

/**
 * this function must be used when loading data from device.
 */
int gdev_callback_load_from_device(void *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = ((struct gdev_handle*)h)->ctx;
	uint32_t fence;

	fence = gdev_memcpy(ctx, dst_addr, src_addr, size);
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
	struct gdev_handle *h = NULL;
	struct gdev_device *gdev = NULL;
	struct gdev_sched_entity *se = NULL;
	gdev_vas_t *vas = NULL;
	gdev_ctx_t *ctx = NULL;
	gdev_mem_t **dma_mem = NULL;

	if (!(h = MALLOC(sizeof(*h)))) {
		GDEV_PRINT("Failed to allocate device handle\n");
		return NULL;
	}
	memset(h, 0, sizeof(*h));

	h->pipeline_count = GDEV_PIPELINE_DEFAULT_COUNT;
	h->chunk_size = GDEV_CHUNK_DEFAULT_SIZE;

	/* open the specified device. */
	gdev = gdev_dev_open(minor);
	if (!gdev) {
		GDEV_PRINT("Failed to open gdev%d\n", minor);
		goto fail_open;
	}

	/* none can access GPU while someone is opening device. */
	gdev_block_start(gdev);

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

#ifndef GDEV_SCHED_DISABLED
	/* allocate a scheduling entity. */
	se = gdev_sched_entity_create(gdev, ctx);
	if (!se) {
		GDEV_PRINT("Failed to allocate scheduling entity\n");
		goto fail_se;
	}
#endif
	
	/* now other users can access the GPU. */
	gdev_block_end(gdev);

	/* save the objects to the handle. */
	h->se = se;
	h->dma_mem = dma_mem;
	h->vas = vas;
	h->ctx = ctx;
	h->gdev = gdev;
	h->dev_id = minor;

	GDEV_PRINT("Opened gdev%d\n", minor);

	return h;

#ifndef GDEV_SCHED_DISABLED
fail_se:
	__free_dma(dma_mem, h->pipeline_count);
#endif
fail_dma:
	gdev_ctx_free(ctx);
fail_ctx:
	gdev_vas_free(vas);
fail_vas:
	gdev_block_end(gdev);
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
	struct gdev_device *gdev=h->gdev;

	if (!h)
		return -ENOENT;
	if (!h->gdev || !h->ctx || !h->vas)
		return -ENOENT;
	
	/* none can access GPU while someone is closing device. */
	gdev_block_start(gdev);

#ifndef GDEV_SCHED_DISABLED
	if (!h->se)
		return -ENOENT;
	/* free the scheduling entity. */
	gdev_sched_entity_destroy(h->se);
#endif
	
	/* free the bounce buffer. */
	if (h->dma_mem)
		__free_dma(h->dma_mem, h->pipeline_count);

	/* garbage collection: free all memory left in heap. */
	gdev_mem_gc(h->vas);

	/* free the objects. */
	gdev_ctx_free(h->ctx);
	gdev_vas_free(h->vas);
	
	gdev_block_end(gdev);
	
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

	if (gdev->mem_used + size > gdev->mem_size) {
		/* try to share memory with someone (only for device memory). 
		   the shared memory must be freed in gdev_mem_free() when 
		   unreferenced by all users. */
		if (!(mem = gdev_mem_share(vas, size))) {
			GDEV_PRINT("Failed to share memory with victims\n");
			goto fail;
		}
	}
	else if (!(mem = gdev_mem_alloc(vas, size, GDEV_MEM_DEVICE))) {
		if (!(mem = gdev_mem_share(vas, size))) {
			GDEV_PRINT("Failed to share memory with victims\n");
			goto fail;
		}
	}

	return gdev_mem_getaddr(mem);

fail:
	return 0;
}

/**
 * gfree():
 * free the memory space allocated at the specified address.
 */
uint64_t gfree(struct gdev_handle *h, uint64_t addr)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t size;

	if (!(mem = gdev_mem_lookup_by_addr(vas, addr, GDEV_MEM_DEVICE)))
		goto fail;
	size = gdev_mem_getsize(mem);
	gdev_mem_free(mem);

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

	if (gdev->dma_mem_used + size > gdev->dma_mem_size)
		goto fail;
	else if (!(mem = gdev_mem_alloc(vas, size, GDEV_MEM_DMA)))
		goto fail;

	return gdev_mem_getbuf(mem);

fail:
	return 0;
}

/**
 * gfree_dma():
 * free the host dma memory space allocated at the specified buffer.
 */
uint64_t gfree_dma(struct gdev_handle *h, void *buf)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t size;

	if (!(mem = gdev_mem_lookup_by_buf(vas, buf, GDEV_MEM_DMA)))
		goto fail;
	size = gdev_mem_getsize(mem);
	gdev_mem_free(mem);

	return size;

fail:
	return 0;
}

/**
 * gmap():
 * map device memory to host DMA memory.
 */
void *gmap(struct gdev_handle *h, uint64_t addr, uint64_t size)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t offset;
	
	if (!(mem = gdev_mem_lookup_by_addr(vas, addr, GDEV_MEM_DEVICE)))
		goto fail;

	offset = addr - gdev_mem_getaddr(mem);
	return gdev_mem_map(mem, offset, size);

fail:
	return NULL;
}

/**
 * gunmap():
 * unmap device memory from host DMA memory.
 */
int gunmap(struct gdev_handle *h, void *buf)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	
	if (!(mem = gdev_mem_lookup_by_buf(vas, buf, GDEV_MEM_DEVICE)))
		goto fail;

	gdev_mem_unmap(mem);

	return 0;

fail:
	return -ENOENT;
}

/**
 * gmemcpy_to_device():
 * copy data from @buf to device memory at @addr.
 */
int gmemcpy_to_device(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, NULL, __f_memcpy);
}

/**
 * gmemcpy_to_device_async():
 * asynchronously copy data from @buf to device memory at @addr.
 */
int gmemcpy_to_device_async(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id)
{
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, id, __f_memcpy);
}

/**
 * gmemcpy_user_to_device():
 * copy data from "user-space" @buf to device memory at @addr.
 */
int gmemcpy_user_to_device(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, NULL, __f_cfu);
}

/**
 * gmemcpy_user_to_device_async():
 * asynchrounouly copy data from "user-space" @buf to device memory at @addr.
 */
int gmemcpy_user_to_device_async(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id)
{
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, id, __f_cfu);
}

/**
 * gmemcpy_from_device():
 * copy data from device memory at @addr to @buf.
 */
int gmemcpy_from_device(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, NULL, __f_memcpy);
}

/**
 * gmemcpy_from_device_async():
 * asynchronously copy data from device memory at @addr to @buf.
 */
int gmemcpy_from_device_async(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id)
{
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, id, __f_memcpy);
}

/**
 * gmemcpy_user_from_device():
 * copy data from device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, NULL, __f_ctu);
}

/**
 * gmemcpy_user_from_device_async():
 * asynchronously copy data from device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device_async(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id)
{
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, id, __f_ctu);
}

/**
 * gmemcpy():
 * copy data of the given size within the global address space.
 * this could be HtoD, DtoH, DtoD, and HtoH.
 */
int gmemcpy(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
#ifndef GDEV_SCHED_DISABLED
	struct gdev_sched_entity *se = h->se;
	struct gdev_device *gdev = h->gdev;
#endif
	gdev_ctx_t *ctx = h->ctx;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *dst;
	gdev_mem_t *src;
	uint32_t fence;

	dst = gdev_mem_lookup_by_addr(vas, dst_addr, GDEV_MEM_DEVICE);
	if (!dst) {
		dst = gdev_mem_lookup_by_addr(vas, dst_addr, GDEV_MEM_DMA);
		if (!dst)
			return -ENOENT;
	}

	src = gdev_mem_lookup_by_addr(vas, src_addr, GDEV_MEM_DEVICE);
	if (!src) {
		src = gdev_mem_lookup_by_addr(vas, src_addr, GDEV_MEM_DMA);
		if (!src)
			return -ENOENT;
	}

#ifndef GDEV_SCHED_DISABLED
	/* decide if the context needs to stall or not. */
	gdev_schedule_memory(se);
#endif

	gdev_mem_lock(dst);
	gdev_mem_lock(src);

	fence = gdev_memcpy(ctx, dst_addr, src_addr, size); 
	gdev_poll(ctx, fence, NULL);

	gdev_mem_unlock(src);
	gdev_mem_unlock(dst);

#ifndef GDEV_SCHED_DISABLED
	/* select the next context by itself, since memcpy is synchronous. */
	gdev_select_next_memory(gdev);
#endif

	return 0;
}

/**
 * gmemcpy_async():
 * asynchronously copy data of the given size within the global address space.
 * this could be HtoD, DtoH, DtoD, and HtoH.
 */
int gmemcpy_async(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size, uint32_t *id)
{
#ifndef GDEV_SCHED_DISABLED
	struct gdev_sched_entity *se = h->se;
	struct gdev_device *gdev = h->gdev;
#endif
	gdev_ctx_t *ctx = h->ctx;
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *dst;
	gdev_mem_t *src;
	uint32_t fence;

	dst = gdev_mem_lookup_by_addr(vas, dst_addr, GDEV_MEM_DEVICE);
	if (!dst) {
		dst = gdev_mem_lookup_by_addr(vas, dst_addr, GDEV_MEM_DMA);
		if (!dst)
			return -ENOENT;
	}

	src = gdev_mem_lookup_by_addr(vas, src_addr, GDEV_MEM_DEVICE);
	if (!src) {
		src = gdev_mem_lookup_by_addr(vas, src_addr, GDEV_MEM_DMA);
		if (!src)
			return -ENOENT;
	}

#ifndef GDEV_SCHED_DISABLED
	/* decide if the context needs to stall or not. */
	gdev_schedule_memory(se);
#endif

	gdev_mem_lock(dst);
	gdev_mem_lock(src);

	fence = gdev_memcpy_async(ctx, dst_addr, src_addr, size); 

	gdev_mem_unlock(src);
	gdev_mem_unlock(dst);

#ifndef GDEV_SCHED_DISABLED
	/* this should be done upon interrupt. */
	gdev_select_next_memory(gdev);
#endif

	*id = fence;

	return 0;
}

/**
 * glaunch():
 * launch the GPU kernel code.
 */
int glaunch(struct gdev_handle *h, struct gdev_kernel *kernel, uint32_t *id)
{
#ifndef GDEV_SCHED_DISABLED
	struct gdev_sched_entity *se = h->se;
#endif
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;

#ifndef GDEV_SCHED_DISABLED
	/* decide if the context needs to stall or not. */
	gdev_schedule_compute(se);
#endif

	gdev_mem_lock_all(vas);

	gdev_shm_retrieve_swap_all(ctx, vas); /* get all data swapped back! */
	*id = gdev_launch(ctx, kernel);

	gdev_mem_unlock_all(vas); /* this should be called when compute done... */

	return 0;
}

/**
 * gsync():
 * poll until the GPU becomes available.
 * @timeout is a unit of milliseconds.
 */
int gsync(struct gdev_handle *h, uint32_t id, struct gdev_time *timeout)
{
	/* @id could be zero if users have called memcpy_async in a wrong way. */
	if (id == 0)
		return 0;
#ifndef __KERNEL__
        int ret = gdev_poll(h->ctx, id, timeout);
#ifndef GDEV_SCHED_DISABLED
        gdev_next_compute(h->gdev);
#endif
        return ret;
#else

        return gdev_poll(h->ctx, id, timeout);
#endif
}

/**
 * gbarrier():
 * explicitly barrier the memory.
 */
int gbarrier(struct gdev_handle *h)
{
	return gdev_barrier(h->ctx);
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

	if (key < 0 || size == 0)
		return -EINVAL;

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
	if (!(new = gdev_shm_attach(vas, owner, gdev_mem_getsize(owner))))
		goto fail;
	gdev_mutex_unlock(&gdev->shm_mutex);

	return gdev_mem_getaddr(new);

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
	if (!(mem = gdev_mem_lookup_by_addr(vas, addr, GDEV_MEM_DEVICE)))
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

/**
 * gref():
 * reference virtual memory from handle @hsrc to handle @hdst.
 */
uint64_t gref(Ghandle hmaster, uint64_t addr, uint64_t size, Ghandle hslave)
{
	gdev_mem_t *mem, *new;
	
	mem = gdev_mem_lookup_by_addr(hmaster->vas, addr, GDEV_MEM_DEVICE);
	if (!mem) {
		/* try to find a host DMA memory object. */
		mem = gdev_mem_lookup_by_addr(hmaster->vas, addr, GDEV_MEM_DMA);
		if (!mem)
			return 0;
	}

	new = gdev_shm_attach(hslave->vas, mem, size);
	if (!new)
		return 0;

	return gdev_mem_getaddr(new);
}

/**
 * gunref():
 * unreference virtual memory from the shared region.
 */
int gunref(Ghandle h, uint64_t addr)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	
	mem = gdev_mem_lookup_by_addr(vas, addr, GDEV_MEM_DEVICE);
	if (!mem) {
		/* try to find a host DMA memory object. */
		mem = gdev_mem_lookup_by_addr(vas, addr, GDEV_MEM_DMA);
		if (!mem)
			return -ENOENT;
	}

	gdev_shm_detach(mem);

	return 0;
}

/**
 * gphysget():
 * get the physical (PCI) bus address associated with buffer pointer @p
 */
uint64_t gphysget(Ghandle h, const void *p)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t offset;
	
	mem = gdev_mem_lookup_by_buf(vas, p, GDEV_MEM_DEVICE);
	if (!mem) {
		mem = gdev_mem_lookup_by_buf(vas, p, GDEV_MEM_DMA);
		if (!mem)
			goto fail;
	}

	offset = (uint64_t)p - (uint64_t)gdev_mem_getbuf(mem);

	return gdev_mem_phys_getaddr(mem, offset);
	
fail:
	return 0;
}

/**
 * gvirtget():
 * get the unified virtual address associated with buffer pointer @p
 */
uint64_t gvirtget(Ghandle h, const void *p)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;
	uint64_t offset;
	
	mem = gdev_mem_lookup_by_buf(vas, p, GDEV_MEM_DEVICE);
	if (!mem) {
		mem = gdev_mem_lookup_by_buf(vas, p, GDEV_MEM_DMA);
		if (!mem)
			goto fail;
	}

	offset = (uint64_t)p - (uint64_t)gdev_mem_getbuf(mem);

	return gdev_mem_getaddr(mem) + offset;
	
fail:
	return 0;
}

/**
 * gdevice_count():
 * get the count of virtual devices
 */
int gdevice_count(int* result)
{
	*result = gdev_getinfo_device_count();
	return 0;
}
