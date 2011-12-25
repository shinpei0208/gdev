/*
 * Copyright 2011 Shinpei Kato
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
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
#include "gdev_conf.h"
#include "gdev_mm.h"
#include "gdev_proto.h"

#define __max(x, y) (x) > (y) ? (x) : (y)
#define __min(x, y) (x) < (y) ? (x) : (y)

/**
 * Gdev handle struct: not visible to outside.
 */
struct gdev_handle {
	struct gdev_device *gdev; /* gdev handle object. */
	gdev_vas_t *vas; /* virtual address space object. */
	gdev_ctx_t *ctx; /* device context object. */
	gdev_mem_t **dma_mem; /* host-side DMA memory object (bounce buffer). */
	uint32_t chunk_size; /* configurable memcpy chunk size. */
	int pipeline_count; /* configurable memcpy pipeline count. */
	int dev_id; /* device ID. */
};

static inline 
gdev_mem_t **__malloc_dma(struct gdev_handle *h, gdev_vas_t *vas, uint64_t size)
{
	gdev_mem_t **dma_mem;
	int i;

	dma_mem = MALLOC(sizeof(*dma_mem) * h->pipeline_count);
	if (!dma_mem)
		return NULL;

	for (i = 0; i < h->pipeline_count; i++) {
		dma_mem[i] = gdev_mem_alloc(vas, size, GDEV_MEM_DMA);
		if (!dma_mem[i])
			return NULL;
	}

	return dma_mem;
}

static inline
void __free_dma(struct gdev_handle *h, gdev_mem_t **dma_mem)
{
	int i;

	for (i = 0; i < h->pipeline_count; i++)
		gdev_mem_free(dma_mem[i]);

	FREE(dma_mem);
}

/**
 * gopen():
 * create a new GPU context on the given device #@devnum.
 */
struct gdev_handle *gopen(int minor)
{
	struct gdev_handle *h;
	struct gdev_device *gdev;
	gdev_vas_t *vas;
	gdev_ctx_t *ctx;
	gdev_mem_t **dma_mem;

	if (!(h = MALLOC(sizeof(*h))))
		return NULL;

	h->pipeline_count = GDEV_PIPELINE_DEFAULT_COUNT;
	h->chunk_size = GDEV_CHUNK_DEFAULT_SIZE;

	/* open the specified device. */
	gdev = gdev_dev_open(minor);
	if (!gdev)
		goto fail_open;

	/* create a new virual address space (VAS) object. */
	vas = gdev_vas_new(gdev, GDEV_VAS_SIZE);
	if (!vas)
		goto fail_vas;

	/* create a new GPU context object. */
	ctx = gdev_ctx_new(gdev, vas);
	if (!ctx)
		goto fail_ctx;

	/* allocate static bounce bound buffer objects. */
	dma_mem = __malloc_dma(h, vas, GDEV_CHUNK_DEFAULT_SIZE);
	if (!dma_mem)
		goto fail_dma;

	h->dma_mem = dma_mem;
	h->vas = vas;
	h->ctx = ctx;
	h->gdev = gdev;
	h->dev_id = minor;

	/* insert the created VAS object to the device VAS list. */
	gdev_vas_list_add(vas);

	GDEV_PRINT("Opened gdev%d.\n", minor);

	return h;

fail_dma:
	GDEV_PRINT("Failed to allocate static DMA buffer object.\n");
	gdev_ctx_free(ctx);
fail_ctx:
	GDEV_PRINT("Failed to create a context object.\n");
	gdev_vas_free(vas);
fail_vas:
	GDEV_PRINT("Failed to create a virtual address space object.\n");
	gdev_dev_close(gdev);
fail_open:
	GDEV_PRINT("Failed to open gdev%d.\n", minor);

	return NULL;
}

/**
 * gclose():
 * destroy the GPU context associated with @handle.
 */
int gclose(struct gdev_handle *h)
{
	struct gdev_device *gdev;
	gdev_vas_t *vas;
	gdev_ctx_t *ctx;
	gdev_mem_t **dma_mem;

	if (!h)
		return -ENOENT;
	gdev = h->gdev;
	if (!gdev)
		return -ENOENT;
	ctx = h->ctx;
	if (!ctx)
		return -ENOENT;
	vas = h->vas;
	if (!vas)
		return -ENOENT;
	dma_mem = h->dma_mem;
	if (!dma_mem)
		return -ENOENT;

	/* delete the VAS object from the device VAS list. */
	gdev_vas_list_del(vas);
	/* free the bounce buffer. */
	__free_dma(h, dma_mem);
	/* garbage collection: free all memory left in heap. */
	gdev_mem_gc(vas);
	/* free the objects. */
	gdev_ctx_free(ctx);
	gdev_vas_free(vas);
	gdev_dev_close(gdev);

	GDEV_PRINT("Closed gdev%d.\n", h->dev_id);

	FREE(h);

	return 0;
}

/**
 * gmalloc():
 * allocate new device memory space.
 */
uint64_t gmalloc(struct gdev_handle *h, uint64_t size)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;

	if (!(mem = gdev_mem_alloc(vas, size, GDEV_MEM_DEVICE))) {
		/* a second chance with shared memory only for device memory. */
		if (!(mem = gdev_shmem_request(vas, NULL, size)))
			goto fail;
	}
	gdev_mem_list_add(mem, GDEV_MEM_DEVICE);

	return GDEV_MEM_ADDR(mem);

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

	if (!(mem = gdev_mem_lookup(vas, addr, GDEV_MEM_DEVICE)))
		goto fail;
	size = GDEV_MEM_SIZE(mem);
	gdev_mem_list_del(mem);
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
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *mem;

	if (!(mem = gdev_mem_alloc(vas, size, GDEV_MEM_DMA)))
		goto fail;
	gdev_mem_list_add(mem, GDEV_MEM_DMA);

	return GDEV_MEM_BUF(mem);

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

	if (!(mem = gdev_mem_lookup(vas, (uint64_t)buf, GDEV_MEM_DMA)))
		goto fail;
	size = GDEV_MEM_SIZE(mem);
	gdev_mem_list_del(mem);
	gdev_mem_free(mem);

	return size;

fail:
	return 0;
}

static
int __memcpy_wrapper(void *dst, const void *src, uint32_t size)
{
	memcpy(dst, src, size);
	return 0;
}

static
int __copy_from_user_wrapper(void *dst, const void *src, uint32_t size)
{
	if (COPY_FROM_USER(dst, src, size))
		return -EFAULT;
	return 0;
}

static
int __copy_to_user_wrapper(void *dst, const void *src, uint32_t size)
{
	if (COPY_TO_USER(dst, src, size))
		return -EFAULT;
	return 0;
}

/**
 * real entity to perform gmemcpy_to_device() with multiple pipelines.
 * @memcpy_host is either memcpy() or copy_from_user().
 */
static
int __gmemcpy_to_device_pipeline
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = h->ctx;
	uint32_t chunk_size = h->chunk_size;
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t dma_size;
	int ret = 0;
	int i;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = h->vas;
	if (!(dma_mem = __dma_alloc(vas, ctx, size)))
		return -ENOMEM;
#else
	dma_mem = h->dma_mem;
#endif

	for (i = 0; i < h->pipeline_count; i++) {
		dma_addr[i] = GDEV_MEM_ADDR(dma_mem[i]);
		dma_buf[i] = GDEV_MEM_BUF(dma_mem[i]);
		fence[i] = 0;
	}

	offset = 0;
	for (;;) {
		for (i = 0; i < h->pipeline_count; i++) {
			dma_size = __min(rest_size, chunk_size);
			rest_size -= dma_size;
			/* HtoH */
			if (fence[i])
				gdev_poll(ctx, fence[i], NULL);
			ret = memcpy_host(dma_buf[i], src_buf + offset, dma_size);
			if (ret)
				goto end;
			/* HtoD */
			fence[i] = 
				gdev_memcpy(ctx, dst_addr + offset, dma_addr[i], dma_size);

			if (rest_size == 0) {
				/* wait for the last fence, and go out! */
				gdev_poll(ctx, fence[i], NULL);
				goto end;
			}
			offset += dma_size;
		}
	}
end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(h, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_to_device().
 * @memcpy_host is either memcpy() or copy_from_user().
 */
static
int __gmemcpy_to_device
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = h->ctx;
	uint32_t chunk_size = h->chunk_size;
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = h->vas;
	if (!(dma_mem = __dma_alloc(vas, ctx, size)))
		return -ENOMEM;
#else
	dma_mem = h->dma_mem;
#endif

	dma_addr[0] = GDEV_MEM_ADDR(dma_mem[0]);
	dma_buf[0] = GDEV_MEM_BUF(dma_mem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, chunk_size);
		ret = memcpy_host(dma_buf[0], src_buf + offset, dma_size);
		if (ret)
			goto end;
		fence = gdev_memcpy(ctx, dst_addr + offset, dma_addr[0], dma_size);
		gdev_poll(ctx, fence, NULL);
		rest_size -= dma_size;
		offset += dma_size;
	}

end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(h, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_to_device() directly with dma memory.
 */
static 
int __gmemcpy_dma_to_device
(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = h->ctx;
	uint32_t fence;

	/* we don't break data into chunks if copying directly from dma memory. */
	fence = gdev_memcpy(ctx, dst_addr, src_addr, size);
	gdev_poll(ctx, fence, NULL);
	
	return 0;
}

/**
 * real entity to perform gmemcpy_from_device() with multiple pipelines.
 * memcpy_host() is either memcpy() or copy_to_user().
 */
static
int __gmemcpy_from_device_pipeline
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = h->ctx;
	uint32_t chunk_size = h->chunk_size;
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t dma_size;
	int ret = 0;
	int i;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = h->vas;
	uint64_t dma_size = __min(size, chunk_size);
	if (!(dma_mem = __malloc_dma(h, vas, dma_size)))
		return -ENOMEM;
#else
	dma_mem = h->dma_mem;
#endif

	for (i = 0; i < h->pipeline_count; i++) {
		dma_addr[i] = GDEV_MEM_ADDR(dma_mem[i]);
		dma_buf[i] = GDEV_MEM_BUF(dma_mem[i]);
		fence[i] = 0;
	}

	offset = 0;
	dma_size = __min(rest_size, chunk_size);
	rest_size -= dma_size;
	/* DtoH */
	fence[0] = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, dma_size);
	for (;;) {
		for (i = 0; i < h->pipeline_count; i++) {
			if (rest_size == 0) {
				/* HtoH */
				gdev_poll(ctx, fence[i], NULL);
				memcpy_host(dst_buf + offset, dma_buf[i], dma_size);
				goto end;
			}

			dma_size = __min(rest_size, chunk_size);
			rest_size -= dma_size;
			offset += dma_size;

			/* DtoH */
			if (i + 1 == h->pipeline_count) {
				fence[0] = gdev_memcpy(ctx, dma_addr[0], 
									   src_addr + offset, dma_size);
			}
			else {
				fence[i + 1] = gdev_memcpy(ctx, dma_addr[i + 1], 
										   src_addr + offset, dma_size);
			}

			/* HtoH */
			gdev_poll(ctx, fence[i], NULL);
			ret = memcpy_host(dst_buf + offset - chunk_size, dma_buf[i], 
							  chunk_size);
			if (ret)
				goto end;
		}
	}
end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(h, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_from_device().
 * memcpy_host() is either memcpy() or copy_to_user().
 */
static
int __gmemcpy_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = h->ctx;
	uint32_t chunk_size = h->chunk_size;
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = h->vas;
	uint64_t dma_size = __min(size, chunk_size);
	if (!(dma_mem = __malloc_dma(h, vas, dma_size)))
		return -ENOMEM;
#else
	dma_mem = h->dma_mem;
#endif

	dma_addr[0] = GDEV_MEM_ADDR(dma_mem[0]);
	dma_buf[0] = GDEV_MEM_BUF(dma_mem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, chunk_size);
		fence = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, dma_size);
		gdev_poll(ctx, fence, NULL);
		ret = memcpy_host(dst_buf + offset, dma_buf[0], dma_size);
		if (ret)
			goto end;

		rest_size -= dma_size;
		offset += dma_size;
	}
end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(h, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_from_device() with dma memory.
 */
static
int __gmemcpy_dma_from_device
(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = h->ctx;
	uint32_t fence;

	/* we don't break data into chunks if copying directly from dma memory. */
	fence = gdev_memcpy(ctx, dst_addr, src_addr, size); 
	gdev_poll(ctx, fence, NULL);

	return 0;
}

/**
 * gmemcpy_to_device():
 * copy data from @buf to the device memory at @addr.
 */
int gmemcpy_to_device
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *hmem = gdev_mem_lookup(vas, (uint64_t)src_buf, GDEV_MEM_DMA);
	gdev_mem_t *mem = gdev_mem_lookup(vas, dst_addr, GDEV_MEM_DEVICE);
	int ret;

	gdev_shmem_lock(mem);
	if (hmem) 
		ret = __gmemcpy_dma_to_device(h, dst_addr, hmem->addr, size);
	else {
		/* the function will evict data *only if* necessary. */
		gdev_shmem_evict(mem, h);
		if (h->pipeline_count > 1)
			ret =  __gmemcpy_to_device_pipeline(h, dst_addr, src_buf, size,
												__memcpy_wrapper);
		else
			ret = __gmemcpy_to_device(h, dst_addr, src_buf, size,
									  __memcpy_wrapper);
	}
	gdev_shmem_unlock(mem);
	
	return ret;
}

/**
 * gmemcpy_user_to_device():
 * copy data from "user-space" @buf to the device memory at @addr.
 */
int gmemcpy_user_to_device
(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *hmem = gdev_mem_lookup(vas, (uint64_t)src_buf, GDEV_MEM_DMA);
	gdev_mem_t *mem = gdev_mem_lookup(vas, dst_addr, GDEV_MEM_DEVICE);
	int ret;

	gdev_shmem_lock(mem);
	if (hmem)
		ret = __gmemcpy_dma_to_device(h, dst_addr, hmem->addr, size);
	else {
		/* the function will evict data *only if* necessary. */
		gdev_shmem_evict(mem, h);
		if (h->pipeline_count > 1)
			ret = __gmemcpy_to_device_pipeline(h, dst_addr, src_buf, size, 
												__copy_from_user_wrapper);
		else
			ret = __gmemcpy_to_device(h, dst_addr, src_buf, size, 
									   __copy_from_user_wrapper);
	}
	gdev_shmem_unlock(mem);

	return ret;
}

/**
 * gmemcpy_from_device():
 * copy data from the device memory at @addr to @buf.
 */
int gmemcpy_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *hmem = gdev_mem_lookup(vas, (uint64_t)dst_buf, GDEV_MEM_DMA);
	gdev_mem_t *mem = gdev_mem_lookup(vas, src_addr, GDEV_MEM_DEVICE);
	int ret;

	gdev_shmem_lock(mem);
	if (hmem)
		ret = __gmemcpy_dma_from_device(h, hmem->addr, src_addr, size);
	else {
		/* the function will reload data *only if* necessary. */
		gdev_mem_reload(mem, h);
		if (h->pipeline_count > 1)
			ret = __gmemcpy_from_device_pipeline(h, dst_buf, src_addr, size, 
												  __memcpy_wrapper);
		else
			ret = __gmemcpy_from_device(h, dst_buf, src_addr, size, 
										 __memcpy_wrapper);
	}
	gdev_shmem_unlock(mem);

	return ret;
}

/**
 * gmemcpy_user_from_device():
 * copy data from the device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device
(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_vas_t *vas = h->vas;
	gdev_mem_t *hmem = gdev_mem_lookup(vas, (uint64_t)dst_buf, GDEV_MEM_DMA);
	gdev_mem_t *mem = gdev_mem_lookup(vas, src_addr, GDEV_MEM_DEVICE);
	int ret;

	gdev_shmem_lock(mem);
	if (hmem)
		ret = __gmemcpy_dma_from_device(h, hmem->addr, src_addr, size);
	else {
		/* the function will reload data *only if* necessary. */
		gdev_mem_reload(mem, h);
		if (h->pipeline_count > 1)
			ret = __gmemcpy_from_device_pipeline(h, dst_buf, src_addr, size, 
												  __copy_to_user_wrapper);
		else
			ret = __gmemcpy_from_device(h, dst_buf, src_addr, size, 
										 __copy_to_user_wrapper);
	}
	gdev_shmem_unlock(mem);

	return ret;
}

/**
 * gmemcpy_in_device():
 * copy data of the given size within the device memory.
 */
int gmemcpy_in_device
(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	/* to be implemented */
	return 0;
}

/**
 * glaunch():
 * launch the GPU kernel.
 */
int glaunch(struct gdev_handle *h, struct gdev_kernel *kernel, uint32_t *id)
{
	gdev_vas_t *vas = h->vas;
	gdev_ctx_t *ctx = h->ctx;

	gdev_shmem_lock_all(vas);
	*id = gdev_launch(ctx, kernel);
	gdev_shmem_unlock_all(vas);

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
	gdev_vas_t *vas = h->vas;
	gdev_mem_t **dma_mem = h->dma_mem;

	switch (type) {
	case GDEV_TUNE_MEMCPY_PIPELINE_COUNT:
		if (value > GDEV_PIPELINE_MAX_COUNT || 
			value < GDEV_PIPELINE_MIN_COUNT) {
			return -EINVAL;
		}
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		__free_dma(h, dma_mem);
#endif
		/* change the pipeline count here. */
		h->pipeline_count = value;
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		dma_mem = __malloc_dma(h, vas, h->chunk_size);
		if (!dma_mem)
			return -ENOMEM;
		h->dma_mem = dma_mem;
#endif
		break;
	case GDEV_TUNE_MEMCPY_CHUNK_SIZE:
		if (value > GDEV_CHUNK_MAX_SIZE) {
			return -EINVAL;
		}
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		__free_dma(h, dma_mem);
#endif
		/* change the chunk size here. */
		h->chunk_size = value;
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		dma_mem = __malloc_dma(h, vas, h->chunk_size);
		if (!dma_mem)
			return -ENOMEM;
		h->dma_mem = dma_mem;
#endif
		break;
	default:
		return -EINVAL;
	}
	return 0;
}
