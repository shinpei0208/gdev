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

static inline uint64_t min_u64(uint64_t x, uint64_t y)
{
	if (x > y) 
		return y;
	else 
		return x;
}

static inline gdev_mem_t **__malloc_dma
(gdev_handle_t *handle, gdev_vas_t *vas, uint64_t size)
{
	gdev_mem_t **dma_mem;
	int pipelines = GDEV_PIPELINE_GET(handle);
	int i;

	dma_mem = MALLOC(sizeof(*dma_mem) * pipelines);
	if (!dma_mem)
		return NULL;

	for (i = 0; i < pipelines; i++) {
		dma_mem[i] = gdev_malloc_dma(vas, size);
		if (!dma_mem[i])
			return NULL;
	}

	return dma_mem;
}

static inline void __free_dma(gdev_handle_t *handle, gdev_mem_t **dma_mem)
{
	int i;
	int pipelines = GDEV_PIPELINE_GET(handle);

	for (i = 0; i < pipelines; i++)
		gdev_free_dma(dma_mem[i]);

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

	if (!(mem = gdev_malloc_device(vas, size))) {
		GDEV_PRINT("Failed to allocate memory.\n");
		return 0;
	}
	
	gdev_heap_add(mem);

	return GDEV_MEM_ADDR(mem);
}

/**
 * gfree():
 * free the memory space allocated at the specified address.
 */
void gfree(gdev_handle_t *handle, uint64_t addr)
{
	gdev_mem_t *mem;
	gdev_vas_t *vas = GDEV_VAS_GET(handle);

	if ((mem = gdev_heap_lookup(vas, addr))) {
		gdev_heap_del(mem);
		gdev_free_device(mem);
	}
}

static inline int __wrapper_memcpy(void *dst, void *src, uint32_t size)
{
	memcpy(dst, src, size);
	return 0;
}

static inline int __wrapper_copy_from_user(void *dst, void *src, uint32_t size)
{
	if (COPY_FROM_USER(dst, src, size))
		return -EFAULT;
	return 0;
}

static inline int __wrapper_copy_to_user(void *dst, void *src, uint32_t size)
{
	if (COPY_TO_USER(dst, src, size))
		return -EFAULT;
	return 0;
}

/**
 * real entity to perform gmemcpy_from_device() with multiple pipelines.
 * @memcpy_host() is either memcpy() or copy_to_user().
 */
static inline int __gmemcpy_from_device_pipeline
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int (*memcpy_host)(void*, void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	int pipelines = GDEV_PIPELINE_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	uint32_t dma_size;
	int ret = 0;
	int i;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	uint64_t dma_size = min_u64(size, chunk_size);
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
	dma_size = min_u64(rest_size, chunk_size);
	rest_size -= dma_size;
	/* DtoH */
	fence[0] = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, dma_size);
	for (;;) {
		for (i = 0; i < pipelines; i++) {
			if (rest_size == 0) {
				/* HtoH */
				gdev_poll(ctx, GDEV_FENCE_DMA, fence[i]);
				memcpy_host(dst_buf + offset, dma_buf[i], dma_size);
				goto end;
			}

			dma_size = min_u64(rest_size, chunk_size);
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
			gdev_poll(ctx, GDEV_FENCE_DMA, fence[i]);
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
 * real entity to perform gmemcpy_from_device() synchronously.
 * @memcpy_host() is either memcpy() or copy_to_user().
 */
static inline int __gmemcpy_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size, 
 int (*memcpy_host)(void*, void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	uint32_t fence;
	uint32_t dma_size;
	int ret = 0;

#ifdef GDEV_NO_STATIC_BOUNCE_BUFFER
	gdev_vas_t *vas = GDEV_VAS_GET(handle);
	uint64_t dma_size = min_u64(size, chunk_size);
	if (!(dma_mem = __malloc_dma(handle, vas, dma_size)))
		return -ENOMEM;
#else
	dma_mem = GDEV_DMA_GET(handle);
#endif

	dma_addr[0] = GDEV_MEM_ADDR(dma_mem[0]);
	dma_buf[0] = GDEV_MEM_BUF(dma_mem[0]);

	/* copy data by the bounce buffer size. */
	offset = 0;
	while (rest_size) {
		dma_size = min_u64(rest_size, chunk_size);
		fence = gdev_memcpy(ctx, dma_addr[0], src_addr + offset, dma_size);
		gdev_poll(ctx, GDEV_FENCE_DMA, fence);
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
 * real entity to perform gmemcpy_to_device() with multiple pipelines.
 * @memcpy_host() is either memcpy() or copy_from_user().
 */
static inline int __gmemcpy_to_device_pipeline
(gdev_handle_t *handle, uint64_t dst_addr, void *src_buf, uint64_t size, 
 int (*memcpy_host)(void*, void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	int pipelines = GDEV_PIPELINE_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	uint32_t fence[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
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
			dma_size = min_u64(rest_size, chunk_size);
			rest_size -= dma_size;
			/* HtoH */
			if (fence[i])
				gdev_poll(ctx, GDEV_FENCE_DMA, fence[i]);
			ret = memcpy_host(dma_buf[i], src_buf + offset, dma_size);
			if (ret)
				goto end;
			/* HtoD */
			fence[i] = 
				gdev_memcpy(ctx, dst_addr + offset, dma_addr[i], dma_size);

			if (rest_size == 0) {
				/* wait for the last fence, and go out! */
				gdev_poll(ctx, GDEV_FENCE_DMA, fence[i]);
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
 * real entity to perform gmemcpy_to_device() synchrounously.
 * @memcpy_host() is either memcpy() or copy_from_user().
 */
static inline int __gmemcpy_to_device
(gdev_handle_t *handle, uint64_t dst_addr, void *src_buf, uint64_t size, 
 int (*memcpy_host)(void*, void*, uint32_t))
{
	gdev_mem_t **dma_mem;
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);
	uint32_t chunk_size = GDEV_CHUNK_GET(handle);
	uint64_t rest_size = size;
	uint64_t offset;
	uint64_t dma_addr[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
	void *dma_buf[GDEV_PIPELINE_MAX_COUNT]; /* may not be fully used. */
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
	offset = 0;

	/* copy data by the bounce buffer size. */
	while (rest_size) {
		dma_size = min_u64(rest_size, chunk_size);
		ret = memcpy_host(dma_buf[0], src_buf + offset, dma_size);
		if (ret)
			goto end;
		fence = gdev_memcpy(ctx, dst_addr + offset, dma_addr[0], dma_size);
		gdev_poll(ctx, GDEV_FENCE_DMA, fence);
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
 * gmemcpy_from_device():
 * copy data from the device memory at @addr to @buf.
 */
int gmemcpy_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_from_device_pipeline(handle, dst_buf, src_addr, size, 
											  __wrapper_memcpy);
	else
		return __gmemcpy_from_device(handle, dst_buf, src_addr, size, 
									 __wrapper_memcpy);
}

/**
 * gmemcpy_user_from_device():
 * copy data from the device memory at @addr to "user-space" @buf.
 */
int gmemcpy_user_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_from_device_pipeline(handle, dst_buf, src_addr, size, 
											  __wrapper_copy_to_user);
	else
		return __gmemcpy_from_device(handle, dst_buf, src_addr, size, 
									 __wrapper_copy_to_user);
}

/**
 * gmemcpy_to_device():
 * copy data from @buf to the device memory at @addr.
 */
int gmemcpy_to_device
(gdev_handle_t *handle, uint64_t dst_addr, void *src_buf, uint64_t size)
{
	if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_to_device_pipeline(handle, dst_addr, src_buf, size,
										__wrapper_memcpy);
	else
		return __gmemcpy_to_device(handle, dst_addr, src_buf, size,
							   __wrapper_memcpy);
}

/**
 * gmemcpy_user_to_device():
 * copy data from "user-space" @buf to the device memory at @addr.
 */
int gmemcpy_user_to_device
(gdev_handle_t *handle, uint64_t dst_addr, void *src_buf, uint64_t size)
{
	if (GDEV_PIPELINE_GET(handle) > 1)
		return __gmemcpy_to_device_pipeline(handle, dst_addr, src_buf, size, 
											__wrapper_copy_from_user);
	else
		return __gmemcpy_to_device(handle, dst_addr, src_buf, size, 
								   __wrapper_copy_from_user);
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
#define U64 long long unsigned int /* to avoid warnings in user-space */
int glaunch(gdev_handle_t *handle, struct gdev_kernel *kernel, uint32_t *id)
{
#ifdef GDEV_DEBUG
	int i;
	GDEV_PRINT("code_addr = 0x%llx\n", (U64) kernel->code_addr);
	GDEV_PRINT("code_pc = 0x%x\n", kernel->code_pc);
	GDEV_PRINT("cmem_addr = 0x%llx\n", (U64) kernel->cmem_addr);
	GDEV_PRINT("cmem_segment = 0x%x\n", kernel->cmem_segment);
	GDEV_PRINT("cmem_size = 0x%x\n", kernel->cmem_size);
	GDEV_PRINT("lmem_addr = 0x%llx\n", (U64) kernel->lmem_addr);
	GDEV_PRINT("lmem_size_total = 0x%llx\n", (U64) kernel->lmem_size_total);
	GDEV_PRINT("lmem_size = 0x%x\n", kernel->lmem_size);
	GDEV_PRINT("lmem_size_neg = 0x%x\n", kernel->lmem_size_neg);
	GDEV_PRINT("lmem_base = 0x%x\n", kernel->lmem_base);
	GDEV_PRINT("smem_size = 0x%x\n", kernel->smem_size);
	GDEV_PRINT("smem_base = 0x%x\n", kernel->smem_base);
	GDEV_PRINT("param_start = 0x%x\n", kernel->param_start);
	GDEV_PRINT("param_count = 0x%x\n", kernel->param_count);
	for (i = 0; i < kernel->param_count; i++)
			GDEV_PRINT("param_buf[%d] = 0x%x\n", i, kernel->param_buf[i]);
	GDEV_PRINT("stack_level = 0x%x\n", kernel->stack_level);
	GDEV_PRINT("warp_size = 0x%x\n", kernel->warp_size);
	GDEV_PRINT("reg_count = 0x%x\n", kernel->reg_count);
	GDEV_PRINT("bar_count = 0x%x\n", kernel->bar_count);
	GDEV_PRINT("grid_x = 0x%x\n", kernel->grid_x);
	GDEV_PRINT("grid_y = 0x%x\n", kernel->grid_y);
	GDEV_PRINT("grid_z = 0x%x\n", kernel->grid_z);
	GDEV_PRINT("block_x = 0x%x\n", kernel->block_x);
	GDEV_PRINT("block_y = 0x%x\n", kernel->block_y);
	GDEV_PRINT("block_z = 0x%x\n", kernel->block_z);
#endif

	*id = gdev_launch(GDEV_CTX_GET(handle), kernel);

	return 0;
}

/**
 * gsync():
 * poll until the GPU becomes available.
 */
void gsync(gdev_handle_t *handle, uint32_t id)
{
	gdev_ctx_t *ctx = GDEV_CTX_GET(handle);

	gdev_mb(ctx);
	gdev_poll(ctx, GDEV_FENCE_COMPUTE, id);
}

/**
 * gquery():
 * query the device-specific information.
 */
int gquery(gdev_handle_t *handle, uint32_t type, uint32_t *result)
{
	return gdev_info_query(handle->gdev, type, result);
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
