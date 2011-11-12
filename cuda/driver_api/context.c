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
#include "gdev_api.h"
#include "gdev_cuda.h"

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
	int minor = dev;
	struct CUctx_st *ctx;
	gdev_handle_t *handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (minor < 0 || minor >= gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!pctx)
		return CUDA_ERROR_INVALID_VALUE;

	if (!(ctx = (CUcontext)malloc(sizeof(*ctx))))
		return CUDA_ERROR_OUT_OF_MEMORY;

	if (!(handle = gopen(minor))) {
		return CUDA_ERROR_UNKNOWN;
	}

	ctx->gdev_handle = handle;
	*pctx = ctx;

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
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;
	if (gclose(ctx->gdev_handle))
		return CUDA_ERROR_INVALID_CONTEXT;

	free(ctx);

	return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *device)
{
	printf("cuCtxGetDevice: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
	printf("cuCtxAttach: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuCtxDetach(CUcontext ctx)
{
	printf("cuCtxDetach: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx)
{
	printf("cuCtxPopCurrent: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx)
{
	printf("cuCtxPushCurrent: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void)
{
	printf("cuCtxSynchronize: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

