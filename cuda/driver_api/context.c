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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "cuda.h"
#include "gdev_cuda.h"
#include "gdev_api.h"
#include "gdev_list.h"

struct CUctx_st *gdev_ctx_current = NULL;
struct gdev_list gdev_ctx_list;

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
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	CUresult res;
	struct CUctx_st *ctx;
	struct gdev_cuda_info *cuda_info;
	Ghandle handle;
	int minor = dev;

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

	/* get the CUDA-specific device information. */
	cuda_info = &ctx->cuda_info;
	if (gquery(handle, GDEV_QUERY_CHIPSET, &cuda_info->chipset)) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_query_chipset;
	}
	if (gquery(handle, GDEV_NVIDIA_QUERY_MP_COUNT, &cuda_info->mp_count)) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_query_mp_count;
	}

	/* FIXME: per-thread warp size and active warps */
	switch (cuda_info->chipset) {
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

	/* save the current context to the stack, if necessary. */
	gdev_list_init(&ctx->list_entry, ctx);
	if (gdev_ctx_current) {
		gdev_list_add(&gdev_ctx_current->list_entry, &gdev_ctx_list);		
	}

	/* initialize context synchronization list. */
	gdev_list_init(&ctx->sync_list, NULL);

	/* we will trace size of memory allocated by users and # of kernels. */
	ctx->data_size = 0;
	ctx->launch_id = 0;

	gdev_ctx_current = ctx;	/* set to the current context. */
	*pctx = ctx;

	return CUDA_SUCCESS;

fail_query_mp_count:
fail_query_chipset:
	gclose(handle);
fail_open_gdev:
	FREE(ctx);
fail_malloc_ctx:
	return res;
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
	struct gdev_list *list_head;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;

	/* wait for all on-the-fly kernels. */
	cuCtxSynchronize();

	if (gclose(ctx->gdev_handle))
		return CUDA_ERROR_INVALID_CONTEXT;

	list_head = gdev_list_head(&gdev_ctx_list);
	gdev_ctx_current = gdev_list_container(list_head);
	if (gdev_ctx_current)
		gdev_list_del(&gdev_ctx_current->list_entry);

	FREE(ctx);

	return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *device)
{
	GDEV_PRINT("cuCtxGetDevice: Not Implemented Yet\n");
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
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	/* save the current context to the stack. */
	gdev_list_add(&gdev_ctx_current->list_entry, &gdev_ctx_list);
	/* set @ctx to the current context. */
	gdev_ctx_current = ctx;

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
	struct gdev_list *list_head;
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	*pctx = gdev_ctx_current;
	list_head = gdev_list_head(&gdev_ctx_list);
	gdev_ctx_current = gdev_list_container(list_head);
	if (gdev_ctx_current)
		gdev_list_del(&gdev_ctx_current->list_entry);
	
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
	Ghandle handle;
	struct gdev_cuda_launch *l;
	struct gdev_list *p;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gdev_list_empty(&gdev_ctx_current->sync_list))
		return CUDA_SUCCESS;

	handle = gdev_ctx_current->gdev_handle;

	/* synchronize with all kernels. */
	gdev_list_for_each(l, &gdev_ctx_current->sync_list, list_entry) {
		/* if timeout is required, specify gdev_time value instead of NULL. */
		if (gsync(handle, l->id, NULL))
			return CUDA_ERROR_UNKNOWN;
	}

	/* remove all lists. */
	while ((p = gdev_list_head(&gdev_ctx_current->sync_list))) {
		gdev_list_del(p);
		l = gdev_list_container(p);
		FREE(l);
	}

	return CUDA_SUCCESS;
}

