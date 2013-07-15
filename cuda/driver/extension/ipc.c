/*
 * Copyright (C) 2011 Shinpei Kato
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

#include "../cuda.h"
#include "gdev_api.h"
#include "../gdev_cuda.h"

CUresult cuShmGet(int *ptr, int key, size_t size, int flags)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ptr)
		return CUDA_ERROR_INVALID_VALUE;

	handle = ctx->gdev_handle;
	if ((*ptr = gshmget(handle, key, size, flags)) < 0) {
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	return CUDA_SUCCESS;
}

CUresult cuShmAt(CUdeviceptr *dptr, int id, int flags)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!dptr || id < 0)
		return CUDA_ERROR_INVALID_VALUE;

	handle = ctx->gdev_handle;
	if (!(*dptr = (CUdeviceptr) gshmat(handle, id, 0 /* addr */, flags))) {
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	return CUDA_SUCCESS;
}

CUresult cuShmDt(CUdeviceptr dptr)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!dptr)
		return CUDA_ERROR_INVALID_VALUE;

	/* wait for all kernels to complete - some may be using the memory. */
	cuCtxSynchronize();

	handle = ctx->gdev_handle;

	if (gshmdt(handle, (uint64_t)dptr))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

CUresult cuShmCtl(int id, int cmd, void *buf /* FIXME */)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (id < 0 || cmd < 0)
		return CUDA_ERROR_INVALID_VALUE;

	/* wait for all kernels to complete - some may be using the memory. */
	cuCtxSynchronize();

	handle = ctx->gdev_handle;

	if (gshmctl(handle, id, cmd, buf))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}
