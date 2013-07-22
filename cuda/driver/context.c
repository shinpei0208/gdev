/*
 * Copyright (C) 2011 Shinpei Kato
 *
 * Systems Research Lab, University of California at Santa Cruz
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

#include "cuda.h"
#include "gdev_cuda.h"
#include "gdev_api.h"
#include "gdev_list.h"

struct gdev_list gdev_ctx_list;
LOCK_T gdev_ctx_list_lock;

/**
 * Creates a new CUDA context and associates it with the calling thread. 
 * The flags parameter is described below. The context is created with a 
 * usage count of 1 and the caller of cuCtxCreate() must call cuCtxDestroy()
 * or cuCtxDetach() when done using the context. If a context is already 
 * current to the thread, it is supplanted by the newly created context and 
 * may be restored by a subsequent call to cuCtxPopCurrent().
 *
 * The two LSBs of the flags parameter can be used to control how the OS 
 * thread, which owns the CUDA context at the time of an API call, interacts 
 * with the OS scheduler when waiting for results from the GPU.
 *
 * CU_CTX_SCHED_AUTO:
 * The default value if the flags parameter is zero, uses a heuristic based 
 * on the number of active CUDA contexts in the process C and the number of 
 * logical processors in the system P. If C > P, then CUDA will yield to 
 * other OS threads when waiting for the GPU, otherwise CUDA will not yield 
 * while waiting for results and actively spin on the processor.
 *
 * CU_CTX_SCHED_SPIN:
 * Instruct CUDA to actively spin when waiting for results from the GPU. 
 * This can decrease latency when waiting for the GPU, but may lower the 
 * performance of CPU threads if they are performing work in parallel with 
 * the CUDA thread.
 *
 * CU_CTX_SCHED_YIELD:
 * Instruct CUDA to yield its thread when waiting for results from the GPU. 
 * This can increase latency when waiting for the GPU, but can increase the 
 * performance of CPU threads performing work in parallel with the GPU.
 *
 * CU_CTX_BLOCKING_SYNC:
 * Instruct CUDA to block the CPU thread on a synchronization primitive when 
 * waiting for the GPU to finish work.
 *
 * CU_CTX_MAP_HOST:
 * Instruct CUDA to support mapped pinned allocations. This flag must be set
 * in order to allocate pinned host memory that is accessible to the GPU.
 *
 * Note to Linux users:
 *
 * Context creation will fail with CUDA_ERROR_UNKNOWN if the compute mode of
 * the device is CU_COMPUTEMODE_PROHIBITED. Similarly, context creation will 
 * also fail with CUDA_ERROR_UNKNOWN if the compute mode for the device is 
 * set to CU_COMPUTEMODE_EXCLUSIVE and there is already an active context on 
 * the device. The function cuDeviceGetAttribute() can be used with 
 * CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode of the 
 * device. The nvidia-smi tool can be used to set the compute mode for devices.
 * Documentation for nvidia-smi can be obtained by passing a -h option to it.
 *
 * Parameters:
 * pctx - Returned context handle of the new context
 * flags - Context creation flags
 * dev - Device to create context on
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE, 
 * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN 
 */
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	CUresult res;
	struct CUctx_st *ctx;
	struct gdev_cuda_info *cuda_info;
	Ghandle handle;
	int minor = (int)dev;
	int mp_count;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (minor < 0 || minor >= gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!pctx)
		return CUDA_ERROR_INVALID_VALUE;

	if (!(ctx = (CUcontext)MALLOC(sizeof(*ctx)))) {
		res = CUDA_ERROR_OUT_OF_MEMORY;
		goto fail_malloc_ctx;
	}

	if (!(handle = gopen(minor))) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_open_gdev;
	}

	/* save the Gdev handle. */
	ctx->gdev_handle = handle;

	/* save the current context to the stack, if necessary. */
	gdev_list_init(&ctx->list_entry, ctx);

	/* initialize context synchronization list. */
	gdev_list_init(&ctx->sync_list, NULL);
	/* initialize context event list. */
	gdev_list_init(&ctx->event_list, NULL);

	/* we will trace # of kernels. */
	ctx->launch_id = 0;
	/* save the device ID. */
	ctx->minor = minor;

	ctx->flags = flags;
	ctx->usage = 0;
	ctx->destroyed = 0;
	ctx->owner = GETTID();
	ctx->user = 0;

	/* set to the current context. */
	res = cuCtxPushCurrent(ctx);
	if (res != CUDA_SUCCESS)
		goto fail_push_current;

	/* get the CUDA-specific device information. */
	cuda_info = &ctx->cuda_info;
	if (gquery(handle, GDEV_QUERY_CHIPSET, &cuda_info->chipset)) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_query_chipset;
	}
#if 0
	if (gquery(handle, GDEV_NVIDIA_QUERY_MP_COUNT, &cuda_info->mp_count)) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_query_mp_count;
	}
#else
	if ((res = cuDeviceGetAttribute(&mp_count,
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev))
		!= CUDA_SUCCESS) {
		goto fail_query_mp_count;
	}
	cuda_info->mp_count = mp_count;
#endif

	/* FIXME: per-thread warp size and active warps */
	switch (cuda_info->chipset & 0xf0) {
	case 0xc0:
		cuda_info->warp_count = 48;
		cuda_info->warp_size = 32;
		break;
	case 0x50:
		cuda_info->warp_count = 32;
		cuda_info->warp_size = 32;
		break;
	default:
		cuda_info->warp_count = 48;
		cuda_info->warp_size = 32;
	}

	*pctx = ctx;

	return CUDA_SUCCESS;

fail_query_mp_count:
fail_query_chipset:
	cuCtxPopCurrent(&ctx);
fail_push_current:
	gclose(handle);
fail_open_gdev:
	FREE(ctx);
fail_malloc_ctx:
	return res;
}
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	return cuCtxCreate_v2(pctx, flags, dev);
}

static int freeDestroyedContext(CUcontext ctx)
{
	if (ctx->usage > 0)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gclose(ctx->gdev_handle))
		return CUDA_ERROR_INVALID_CONTEXT;

	FREE(ctx);

	return CUDA_SUCCESS;
}

/**
 * Destroys the CUDA context specified by ctx. If the context usage count is 
 * not equal to 1, or the context is current to any CPU thread other than the
 * current one, this function fails. Floating contexts (detached from a CPU 
 * thread via cuCtxPopCurrent()) may be destroyed by this function.
 *
 * Parameters:
 * ctx - Context to destroy
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuCtxDestroy(CUcontext ctx)
{
	struct CUctx_st *cur = NULL;
	CUresult res;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;
	if (cur == ctx) {
		res = cuCtxPopCurrent(&cur);
		if (res != CUDA_SUCCESS)
			return res;
	}

	ctx->destroyed = 1;

	if (cur)
		return freeDestroyedContext(cur);

	return CUDA_SUCCESS;
}

CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
	GDEV_PRINT("cuCtxAttach: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuCtxDetach(CUcontext ctx)
{
	GDEV_PRINT("cuCtxDetach: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Returns a version number in version corresponding to the capabilities of
 * the context (e.g. 3010 or 3020), which library developers can use to direct
 * callers to a specific API version. If ctx is NULL, returns the API version
 * used to create the currently bound context.
 *
 * Note that new API versions are only introduced when context capabilities
 * are changed that break binary compatibility, so the API version and driver
 * version may be different. For example, it is valid for the API version
 * to be 3020 while the driver version is 4010.
 *
 * Parameters:
 *     	ctx 	- Context to check
 *     	version	- Pointer to version
 *
 * Returns:
 *     	CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     	CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_UNKNOWN 
 *
 * Note:
 *     	Note that this function may also return error codes from previous,
 *     	asynchronous launches.
 *
 * See also:
 *     	cuCtxCreate, cuCtxDestroy, cuCtxGetDevice, cuCtxGetLimit,
 *     	cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
 *     	cuCtxSynchronize 
 */
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
	*version = 3020; /* FIXME */

	return CUDA_SUCCESS;
}

/**
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this function returns through pconfig the preferred cache
 * configuration for the current context. This is only a preference.
 * The driver will use the requested configuration if possible, but it is
 * free to choose a different configuration if required to execute functions.
 *
 * This will return a pconfig of CU_FUNC_CACHE_PREFER_NONE on devices where
 * the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 *
 *     CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1
 *                                (default)
 *     CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller
 *                                  L1 cache
 *     CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 *     CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * Parameters:
 *     pconfig 	- Returned cache configuration
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetDevice,
 *     cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig,
 *     cuCtxSetLimit, cuCtxSynchronize, cuFuncSetCacheConfig 
 */
CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pconfig)
		return CUDA_ERROR_INVALID_VALUE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	*pconfig = ctx->config;

	return CUDA_SUCCESS;
}

/**
 * Returns in *pctx the CUDA context bound to the calling CPU thread.
 * If no context is bound to the calling CPU thread then *pctx is set to NULL
 * and CUDA_SUCCESS is returned.
 *
 * Parameters:
 *     	pctx 	- Returned context handle
 *
 * Returns:
 *     	CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 *
 * Note:
 *     	Note that this function may also return error codes from previous,
 *     	asynchronous launches.
 *
 * See also:
 *     	cuCtxSetCurrent, cuCtxCreate, cuCtxDestroy 
 */
CUresult cuCtxGetCurrent(CUcontext *pctx)
{
	struct CUctx_st *ctx = NULL;
	CUresult res;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	LOCK(&gdev_ctx_list_lock);

	gdev_list_for_each(ctx, &gdev_ctx_list, list_entry) {
		if (ctx->user == GETTID())
			break;
	}

	UNLOCK(&gdev_ctx_list_lock);

	if (ctx && ctx->destroyed) {
		res = cuCtxPopCurrent(&ctx);
		if (res != CUDA_SUCCESS)
			return res;
		res = freeDestroyedContext(ctx);
		if (res != CUDA_SUCCESS)
			return res;
		*pctx = NULL;
		return CUDA_ERROR_CONTEXT_IS_DESTROYED;
	}

	*pctx = ctx;

	return CUDA_SUCCESS;
}

/**
 * Returns in *device the ordinal of the current context's device.
 *
 * Parameters:
 *     	device 	- Returned device ID for the current context
 *
 * Returns:
 *     	CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     	CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 *
 * Note:
 *     	Note that this function may also return error codes from previous,
 *     	asynchronous launches.
 *
 * See also:
 *     	cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
 *     	cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig,
 *     	cuCtxSetLimit, cuCtxSynchronize 
 */
CUresult cuCtxGetDevice(CUdevice *device)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!device)
		return CUDA_ERROR_INVALID_VALUE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	res = cuDeviceGet(device, ctx->minor);

	return res;
}

/**
 * Returns in *pvalue the current size of limit. The supported CUlimit values
 * are:
 *
 *     CU_LIMIT_STACK_SIZE: stack size of each GPU thread;
 *     CU_LIMIT_PRINTF_FIFO_SIZE: size of the FIFO used by the printf()
 *                                device system call.
 *     CU_LIMIT_MALLOC_HEAP_SIZE: size of the heap used by the malloc()
 *                                and free() device system calls;
 *
 * Parameters:
 *     limit 	- Limit to query
 *     pvalue 	- Returned size in bytes of limit
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNSUPPORTED_LIMIT 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
 *     cuCtxGetDevice, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig,
 *     cuCtxSetLimit, cuCtxSynchronize 
 */
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pvalue)
		return CUDA_ERROR_INVALID_VALUE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	GDEV_PRINT("cuCtxGetLimit: Not Implemented Yet\n");

	return CUDA_SUCCESS;
}

/**
 * Pushes the given context @ctx onto the CPU thread's stack of current 
 * contexts. The specified context becomes the CPU thread's current context, 
 * so all CUDA functions that operate on the current context are affected.
 *
 * The previous current context may be made current again by calling 
 * cuCtxDestroy() or cuCtxPopCurrent().
 *
 * The context must be "floating," i.e. not attached to any thread. Contexts 
 * are made to float by calling cuCtxPopCurrent().
 *
 * Parameters:
 * ctx - Floating context to attach
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuCtxPushCurrent(CUcontext ctx)
{
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;
	if (ctx->usage)
		return CUDA_ERROR_INVALID_CONTEXT;

	LOCK(&gdev_ctx_list_lock);

	/* save the current context to the stack. */
	ctx->usage++;
	ctx->user = GETTID();
	gdev_list_add(&ctx->list_entry, &gdev_ctx_list);

	UNLOCK(&gdev_ctx_list_lock);

	return CUDA_SUCCESS;
}

/**
 * Pops the current CUDA context from the CPU thread. The CUDA context must 
 * have a usage count of 1. CUDA contexts have a usage count of 1 upon 
 * creation; the usage count may be incremented with cuCtxAttach() and 
 * decremented with cuCtxDetach().
 *
 * If successful, cuCtxPopCurrent() passes back the new context handle in 
 * @pctx. The old context may then be made current to a different CPU thread 
 * by calling cuCtxPushCurrent().
 *
 * Floating contexts may be destroyed by calling cuCtxDestroy().
 *
 * If a context was current to the CPU thread before cuCtxCreate() or 
 * cuCtxPushCurrent() was called, this function makes that context current to
 * the CPU thread again.
 *
 * Parameters:
 * pctx - Returned new context handle
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT 
 */
CUresult cuCtxPopCurrent(CUcontext *pctx)
{
	struct CUctx_st *cur = NULL;
	CUresult res;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;
	if (!cur)
		return CUDA_ERROR_INVALID_CONTEXT;

	/* wait for all on-the-fly kernels. */
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS)
		return res;

	LOCK(&gdev_ctx_list_lock);

	gdev_list_del(&cur->list_entry);
	cur->usage--;
	cur->user = 0;

	UNLOCK(&gdev_ctx_list_lock);

	if (cur->destroyed) {
		res = freeDestroyedContext(cur);
		if (res != CUDA_SUCCESS)
			return res;
		*pctx = NULL;
		return CUDA_ERROR_CONTEXT_IS_DESTROYED;
	}

	*pctx = cur;
	
	return CUDA_SUCCESS;
}

/**
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through config the preferred cache configuration
 * for the current context. This is only a preference.
 * The driver will use the requested configuration if possible, but it is free
 * to choose a different configuration if required to execute the function.
 * Any function preference set via cuFuncSetCacheConfig() will be preferred
 * over this context-wide setting. Setting the context-wide cache configuration
 * to CU_FUNC_CACHE_PREFER_NONE will cause subsequent kernel launches to prefer
 * to not change the cache configuration unless required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 *
 *     CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1
 *                                (default)
 *     CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller
 *                                  L1 cache
 *     CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 *     CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * Parameters:
 *     config 	- Requested cache configuration
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
 *     cuCtxGetDevice, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent,
 *     cuCtxSetLimit, cuCtxSynchronize, cuFuncSetCacheConfig 
 */
CUresult cuCtxSetCacheConfig(CUfunc_cache config)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	ctx->config = config;

	return CUDA_SUCCESS;
}

/**
 * Binds the specified CUDA context to the calling CPU thread. If ctx is NULL
 * then the CUDA context previously bound to the calling CPU thread is unbound
 * and CUDA_SUCCESS is returned.
 *
 * If there exists a CUDA context stack on the calling CPU thread, this will
 * replace the top of that stack with ctx. If ctx is NULL then this will be
 * equivalent to popping the top of the calling CPU thread's CUDA context stack
 * (or a no-op if the calling CPU thread's CUDA context stack is empty).
 *
 * Parameters:
 *     ctx 	- Context to bind to the calling CPU thread
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuCtxGetCurrent, cuCtxCreate, cuCtxDestroy 
 */
CUresult cuCtxSetCurrent(CUcontext ctx)
{
	CUresult res;
	CUcontext cur;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxPopCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (ctx)
		res = cuCtxPushCurrent(ctx);

	return res;
}

/**
 * Setting limit to value is a request by the application to update the
 * current limit maintained by the context. The driver is free to modify
 * the requested value to meet h/w requirements (this could be clamping to
 * minimum or maximum values, rounding up to nearest element size, etc).
 * The application can use cuCtxGetLimit() to find out exactly what the
 * limit has been set to.
 *
 * Setting each CUlimit has its own specific restrictions, so each is
 * discussed here.
 *
 *     CU_LIMIT_STACK_SIZE controls the stack size of each GPU thread.
 *     This limit is only applicable to devices of compute capability 2.0 and
 *     higher. Attempting to set this limit on devices of compute capability
 *     less than 2.0 will result in the error CUDA_ERROR_UNSUPPORTED_LIMIT
 *     being returned.
 *
 *     CU_LIMIT_PRINTF_FIFO_SIZE controls the size of the FIFO used by the
 *     printf() device system call.
 *     Setting CU_LIMIT_PRINTF_FIFO_SIZE must be performed before launching
 *     any kernel that uses the printf() device system call, otherwise
 *     CUDA_ERROR_INVALID_VALUE will be returned.
 *     This limit is only applicable to devices of compute capability 2.0 and
 *     higher. Attempting to set this limit on devices of compute capability
 *     less than 2.0 will result in the error CUDA_ERROR_UNSUPPORTED_LIMIT
 *     being returned.
 *
 *     CU_LIMIT_MALLOC_HEAP_SIZE controls the size of the heap used by the
 *     malloc() and free() device system calls.
 *     Setting CU_LIMIT_MALLOC_HEAP_SIZE must be performed before launching
 *     any kernel that uses the malloc() or free() device system calls,
 *     otherwise CUDA_ERROR_INVALID_VALUE will be returned.
 *     This limit is only applicable to devices of compute capability 2.0 and
 *     higher. Attempting to set this limit on devices of compute capability
 *     less than 2.0 will result in the error CUDA_ERROR_UNSUPPORTED_LIMIT
 *     being returned.
 *
 * Parameters:
 *     limit 	- Limit to set
 *     value 	- Size in bytes of limit
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNSUPPORTED_LIMIT 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
 *     cuCtxGetDevice, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent,
 *     cuCtxSetCacheConfig, cuCtxSynchronize 
 */
CUresult cuCtxSetLimit(CUlimit limit, size_t value)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	GDEV_PRINT("cuCtxSetLimit: Not Implemented Yet\n");

	return CUDA_SUCCESS;
}

/**
 * Blocks until the device has completed all preceding requested tasks. 
 * cuCtxSynchronize() returns an error if one of the preceding tasks failed.
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 * CUDA_ERROR_INVALID_CONTEXT 
 */
CUresult cuCtxSynchronize(void)
{
	struct CUctx_st *cur = NULL;
	CUresult res;
	Ghandle handle;
	struct gdev_cuda_fence *f;
	struct gdev_list *p;
	TIME_T time;
	struct CUevent_st *e;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;
	if (!cur)
		return CUDA_ERROR_INVALID_CONTEXT;

#if 0
	if (gdev_list_empty(&cur->sync_list))
		return CUDA_SUCCESS;
#endif

	handle = cur->gdev_handle;

	/* synchronize with all kernels. */
	gdev_list_for_each(f, &cur->sync_list, list_entry) {
		/* if timeout is required, specify gdev_time value instead of NULL. */
		if (gsync(handle, f->id, NULL))
			return CUDA_ERROR_UNKNOWN;
	}

	/* complete event */
	GETTIME(&time);
	while ((p = gdev_list_head(&cur->event_list))) {
		gdev_list_del(p);
		e = gdev_list_container(p);
		e->time = time;
		e->record = 0;
		e->complete = 1;
	}

	/* remove all lists. */
	while ((p = gdev_list_head(&cur->sync_list))) {
		gdev_list_del(p);
		f = gdev_list_container(p);
		FREE(f);
	}

	if (gbarrier(handle))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

