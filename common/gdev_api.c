/*
 * Copyright 2011 Shinpei Kato
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
#include "gdev_conf.h"

#define __max(x, y) (x) > (y) ? (x) : (y)
#define __min(x, y) (x) < (y) ? (x) : (y)

static inline 
gdev_mem_t **__malloc_dma(gdev_handle_t *handle, gdev_vas_t *vas, uint64_t size)
{
	gdev_mem_t **dma_mem;
	int pipelines = GDEV_PIPELINE_GET(handle);
	int i;

	dma_mem = MALLOC(sizeof(*dma_mem) * pipelines);
	if (!dma_mem)
		return NULL;

	for (i = 0; i < pipelines; i++) {
		dma_mem[i] = gdev_malloc(vas, size, GDEV_MEM_DMA);
		if (!dma_mem[i])
			return NULL;
	}

	return dma_mem;
}

static inline
void __free_dma(gdev_handle_t *handle, gdev_mem_t **dma_mem)
{
	int i;
	int pipelines = GDEV_PIPELINE_GET(handle);

	for (i = 0; i < pipelines; i++)
		gdev_free(dma_mem[i]);

	FREE(dma_mem);
}

/**
 * gopen():
 * create a new GPU context on the given device #@devnum.
 */
gdev_handle_t *gopen(int devnum)
{
	gdev_device_t *gdev;
	gdev_handle_t *handle;
	gdev_vas_t *vas;
	gdev_ctx_t *ctx;
	gdev_mem_t **dma_mem;

	if (!(handle = MALLOC(sizeof(*handle))))
		return NULL;

	GDEV_PIPELINE_SET(handle, GDEV_PIPELINE_DEFAULT_COUNT);
	GDEV_CHUNK_SET(handle, GDEV_CHUNK_DEFAULT_SIZE);

	/* open the specified device. */
	gdev = gdev_dev_open(devnum);
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

	/* initialize the list of memory spaces. */
	gdev_heap_init(vas);
	
	/* allocate static bounce bound buffer objects. */
	dma_mem = __malloc_dma(handle, vas, GDEV_CHUNK_GET(handle));
	if (!dma_mem)
		goto fail_dma;

	GDEV_DMA_SET(handle, dma_mem);
	GDEV_VAS_SET(handle, vas);
	GDEV_CTX_SET(handle, ctx);
	GDEV_DEV_SET(handle, gdev);
	GDEV_MINOR_SET(handle, devnum);

	GDEV_PRINT("Opened gdev%d.\n", devnum);

	return handle;

fail_dma:
	gdev_ctx_free(ctx);
fail_ctx:
	gdev_vas_free(vas);
fail_vas:
	gdev_dev_close(gdev);
fail_open:
	GDEV_PRINT("Failed to open gdev%d.\n", devnum);

	return NULL;
}

/**
 * gclose():
 * destroy the GPU context associated with @handle.
 */
int gclose(gdev_handle_t *handle)
{
	gdev_device_t *gdev;
	gdev_vas_t *vas;
	gdev_ctx_t *ctx;
	gdev_mem_t **dma_mem;

	if (!handle)
		return -ENOENT;
	gdev = GDEV_DEV_GET(handle);
	if (!gdev)
		return -ENOENT;
	ctx = GDEV_CTX_GET(handle);
	if (!ctx)
		return -ENOENT;
	vas = GDEV_VAS_GET(handle);
	if (!vas)
		return -ENOENT;
	dma_mem = GDEV_DMA_GET(handle);
	if (!dma_mem)
		return -ENOENT;

	__free_dma(handle, dma_mem);
	gdev_ctx_free(ctx);
	gdev_vas_free(vas);
	gdev_dev_close(gdev);

	GDEV_PRINT("Closed gdev%d.\n", GDEV_MINOR_GET(handle));

	FREE(handle);

	return 0;
}

/**
 * gmalloc():
 * allocate new device memory space.
 */
uint64_t gmalloc(gdev_handle_t *handle, uint64_t size)
{
	gdev_mem_t *mem;
	gdev_vas_t *vas = GDEV_VAS_GET(handle);

	if (!(mem = gdev_malloc(vas, size, GDEV_MEM_DEVICE))) {
		GDEV_PRINT("Failed to allocate memory.\n");
		return 0;
	}
	
	gdev_heap_add(mem, GDEV_MEM_DEVICE);

	return GDEV_MEM_ADDR(mem);
}

/**
 * gfree():
 * free the memory space allocated at the specified address.
 */
int gfree(gdev_handle_t *handle, uint64_t addr)
{
	gdev_mem_t *mem;
	gdev_vas_t *vas = GDEV_VAS_GET(handle);

	if ((mem = gdev_heap_lookup(vas, addr, GDEV_MEM_DEVICE))) {
		gdev_heap_del(mem);
		gdev_free(mem);
		return 0;
	}

	return -ENOENT;
}

/**
 * gmalloc_dma():
 * allocate new host dma memory space.
 */
void *gmalloc_dma(gdev_handle_t *handle, uint64_t size)
{
	gdev_mem_t *mem;
	gdev_vas_t *vas = GDEV_VAS_GET(handle);

	if (!(mem = gdev_malloc(vas, size, GDEV_MEM_DMA))) {
		GDEV_PRINT("Failed to allocate host DMA memory.\n");
		return 0;
	}
	
	gdev_heap_add(mem, GDEV_MEM_DMA);

	return GDEV_MEM_BUF(mem);
}

/**
 * gfree_dma():
 * free the host dma memory space allocated at the specified buffer.
 */
int gfree_dma(gdev_handle_t *handle, void *buf)
{
	gdev_mem_t *mem;
	gdev_vas_t *vas = GDEV_VAS_GET(handle);

	if ((mem = gdev_heap_lookup(vas, (uint64_t)buf, GDEV_MEM_DMA))) {
		gdev_heap_del(mem);
		gdev_free(mem);
		return 0;
	}

	return -ENOENT;
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
(gdev_handle_t *handle, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	int pipelines = GDEV_PIPELINE_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t dma_size;
	int ret = 0;
	int i;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	if (!(dma_mem = __dma_alloc(vas, ctx, size)))
		return -ENOMEM;
#else
	dma_mem = GDEV_DMA_GET(handle);
#endif

	for (i = 0; i < pipelines; i++) {
		dma_addr[i] = GDEV_MEM_ADDR(dma_mem[i]);
		dma_buf[i] = GDEV_MEM_BUF(dma_mem[i]);
		fence[i] = 0;
	}

	offset = 0;
	for (;;) {
		for (i = 0; i < pipelines; i++) {
			dma_size = __min(rest_size, chunk_size);
			rest_size -= dma_size;
			/* HtoH */
			if (fence[i])
				gdev_poll(ctx, GDEV_FENCE_DMA, fence[i], NULL);
			ret = memcpy_host(dma_buf[i], src_buf + offset, dma_size);
			if (ret)
				goto end;
			/* HtoD */
			fence[i] = 
				gdev_memcpy(ctx, dst_addr + offset, dma_addr[i], dma_size);

			if (rest_size == 0) {
				/* wait for the last fence, and go out! */
				gdev_poll(ctx, GDEV_FENCE_DMA, fence[i], NULL);
				goto end;
			}
			offset += dma_size;
		}
	}
end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(handle, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_to_device().
 * @memcpy_host is either memcpy() or copy_from_user().
 */
static
int __gmemcpy_to_device
(gdev_handle_t *handle, uint64_t dst_addr, const void *src_buf, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	if (!(dma_mem = __dma_alloc(vas, ctx, size)))
		return -ENOMEM;
#else
	dma_mem = GDEV_DMA_GET(handle);
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
		gdev_poll(ctx, GDEV_FENCE_DMA, fence, NULL);
		rest_size -= dma_size;
		offset += dma_size;
	}

end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(handle, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_to_device() directly with dma memory.
 */
static
int __gmemcpy_dma_to_device
(gdev_handle_t *handle, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint32_t fence;
	uint32_t dma_size;

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, chunk_size);
		fence = gdev_memcpy(ctx, dst_addr + offset, src_addr + offset, 
							dma_size);
		gdev_poll(ctx, GDEV_FENCE_DMA, fence, NULL);
		rest_size -= dma_size;
		offset += dma_size;
	}

	return 0;
}

/**
 * real entity to perform gmemcpy_from_device() with multiple pipelines.
 * memcpy_host() is either memcpy() or copy_to_user().
 */
static
int __gmemcpy_from_device_pipeline
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	int pipelines = GDEV_PIPELINE_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t dma_size;
	int ret = 0;
	int i;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	uint64_t dma_size = __min(size, chunk_size);
	if (!(dma_mem = __malloc_dma(handle, vas, dma_size)))
		return -ENOMEM;
#else
	dma_mem = GDEV_DMA_GET(handle);
#endif

	for (i = 0; i < pipelines; i++) {
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
		for (i = 0; i < pipelines; i++) {
			if (rest_size == 0) {
				/* HtoH */
				gdev_poll(ctx, GDEV_FENCE_DMA, fence[i], NULL);
				memcpy_host(dst_buf + offset, dma_buf[i], dma_size);
				goto end;
			}

			dma_size = __min(rest_size, chunk_size);
			rest_size -= dma_size;
			offset += dma_size;

			/* DtoH */
			if (i + 1 == pipelines) {
				fence[0] = gdev_memcpy(ctx, dma_addr[0], 
									   src_addr + offset, dma_size);
			}
			else {
				fence[i + 1] = gdev_memcpy(ctx, dma_addr[i + 1], 
										   src_addr + offset, dma_size);
			}

			/* HtoH */
			gdev_poll(ctx, GDEV_FENCE_DMA, fence[i], NULL);
			ret = memcpy_host(dst_buf + offset - chunk_size, dma_buf[i], 
							  chunk_size);
			if (ret)
				goto end;
		}
	}
end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(handle, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_from_device().
 * memcpy_host() is either memcpy() or copy_to_user().
 */
static
int __gmemcpy_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int (*memcpy_host)(void*, const void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT] = {0};
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT] = {0};
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	uint64_t dma_size = __min(size, chunk_size);
	if (!(dma_mem = __malloc_dma(handle, vas, dma_size)))
		return -ENOMEM;
#else
	dma_mem = GDEV_DMA_GET(handle);
#endif

	dma_addr[0] = GDEV_MEM_ADDR(dma_mem[0]);
	dma_buf[0] = GDEV_MEM_BUF(dma_mem[0]);

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, chunk_size);
		fence = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, dma_size);
		gdev_poll(ctx, GDEV_FENCE_DMA, fence, NULL);
		ret = memcpy_host(dst_buf + offset, dma_buf[0], dma_size);
		if (ret)
			goto end;

		rest_size -= dma_size;
		offset += dma_size;
	}
end:
#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	__free_dma(handle, dma_mem);
#endif

	return ret;
}

/**
 * real entity to perform gmemcpy_from_device() with dma memory.
 */
static
int __gmemcpy_dma_from_device
(gdev_handle_t *handle, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint32_t fence;
	uint32_t dma_size;

	/* copy data by the chunk size. */
	offset = 0;
	while (rest_size) {
		dma_size = __min(rest_size, chunk_size);
		fence = gdev_memcpy(ctx, dst_addr + offset, src_addr + offset, 
							dma_size);
		gdev_poll(ctx, GDEV_FENCE_DMA, fence, NULL);
		rest_size -= dma_size;
		offset += dma_size;
	}

	return 0;
}

/**
 * gmemcpy_to_device():
 * copy data from @buf to the device memory at @addr.
 */
int gmemcpy_to_device
(gdev_handle_t *handle, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	gdev_mem_t *hmem = gdev_heap_lookup(vas, (uint64_t)src_buf, GDEV_MEM_DMA);

	if (hmem)
		return __gmemcpy_dma_to_device(handle, dst_addr, hmem->addr, size);
	else if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_to_device_pipeline(handle, dst_addr, src_buf, size,
											__memcpy_wrapper);
	else
		return __gmemcpy_to_device(handle, dst_addr, src_buf, size,
								   __memcpy_wrapper);
}

/**
 * gmemcpy_user_to_device():
 * copy data from "user-space" @buf to the device memory at @addr.
 */
int gmemcpy_user_to_device
(gdev_handle_t *handle, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	gdev_mem_t *hmem = gdev_heap_lookup(vas, (uint64_t)src_buf, GDEV_MEM_DMA);

	if (hmem)
		return __gmemcpy_dma_to_device(handle, dst_addr, hmem->addr, size);
	else if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_to_device_pipeline(handle, dst_addr, src_buf, size, 
											__copy_from_user_wrapper);
	else
		return __gmemcpy_to_device(handle, dst_addr, src_buf, size, 
								   __copy_from_user_wrapper);
}

/**
 * gmemcpy_from_device():
 * copy data from the device memory at @addr to @buf.
 */
int gmemcpy_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	gdev_mem_t *hmem = gdev_heap_lookup(vas, (uint64_t)dst_buf, GDEV_MEM_DMA);

	if (hmem)
		return __gmemcpy_dma_from_device(handle, hmem->addr, src_addr, size);
	if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_from_device_pipeline(handle, dst_buf, src_addr, size, 
											  __memcpy_wrapper);
	else
		return __gmemcpy_from_device(handle, dst_buf, src_addr, size, 
									 __memcpy_wrapper);
}

/**
 * gmemcpy_user_from_device():
 * copy data from the device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	gdev_mem_t *hmem = gdev_heap_lookup(vas, (uint64_t)dst_buf, GDEV_MEM_DMA);

	if (hmem)
		return __gmemcpy_dma_from_device(handle, hmem->addr, src_addr, size);
	if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_from_device_pipeline(handle, dst_buf, src_addr, size, 
											  __copy_to_user_wrapper);
	else
		return __gmemcpy_from_device(handle, dst_buf, src_addr, size, 
									 __copy_to_user_wrapper);
}

/**
 * gmemcpy_in_device():
 * copy data of the given size within the device memory.
 */
int gmemcpy_in_device
(gdev_handle_t *handle, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	/* to be implemented */
	return 0;
}

/**
 * glaunch():
 * launch the GPU kernel.
 */
int glaunch(gdev_handle_t *handle, struct gdev_kernel *kernel, uint32_t *id)
{
	*id = gdev_launch(GDEV_CTX_GET(handle), kernel);
	return 0;
}

/**
 * gsync():
 * poll until the GPU becomes available.
 * @timeout is a unit of milliseconds.
 */
int gsync(gdev_handle_t *handle, uint32_t id, gdev_time_t *timeout)
{
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);

	gdev_mb(ctx);
	return gdev_poll(ctx, GDEV_FENCE_COMPUTE, id, timeout);
}

/**
 * gquery():
 * query the device-specific information.
 */
int gquery(gdev_handle_t *handle, uint32_t type, uint32_t *result)
{
	return gdev_query(handle->gdev, type, result);
}

/**
 * gtune():
 * tune resource management parameters.
 */
int gtune(gdev_handle_t *handle, uint32_t type, uint32_t value)
{
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	gdev_mem_t **dma_mem = GDEV_DMA_GET(handle);

	switch (type) {
	case GDEV_TUNE_MEMCPY_PIPELINE_COUNT:
		if (value > GDEV_PIPELINE_MAX_COUNT || 
			value < GDEV_PIPELINE_MIN_COUNT) {
			return -EINVAL;
		}
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		__free_dma(handle, dma_mem);
#endif
		/* change the pipeline count here. */
		GDEV_PIPELINE_SET(handle, value);
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		dma_mem = __malloc_dma(handle, vas, GDEV_CHUNK_GET(handle));
		if (!dma_mem)
			return -ENOMEM;
		GDEV_DMA_SET(handle, dma_mem);
#endif
		break;
	case GDEV_TUNE_MEMCPY_CHUNK_SIZE:
		if (value > GDEV_CHUNK_MAX_SIZE) {
			return -EINVAL;
		}
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		__free_dma(handle, dma_mem);
#endif
		/* change the chunk size here. */
		GDEV_CHUNK_SET(handle, value);
#ifndef GDEV_NO_STATIC_BOUNCE_BUFFER
		dma_mem = __malloc_dma(handle, vas, GDEV_CHUNK_GET(handle));
		if (!dma_mem)
			return -ENOMEM;
		GDEV_DMA_SET(handle, dma_mem);
#endif
		break;
	default:
		return -EINVAL;
	}
	return 0;
}
