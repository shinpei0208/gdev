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

#include "../cuda.h"
#include "gdev_api.h"
#include "../gdev_cuda.h"


/**
 * Gdev extension: maps device memory to host memory.
 *
 * Parameters:
 * dptr - Device pointer
 * buf - Pointer to user buffer
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemMap(void **buf, CUdeviceptr dptr, unsigned int bytesize)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;
	uint64_t addr = dptr;
	void *map;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!addr || !buf || !bytesize)
		return CUDA_ERROR_INVALID_VALUE;

	handle = ctx->gdev_handle;

	if (!(map = gmap(handle, addr, bytesize)))
		return CUDA_ERROR_UNKNOWN;

	*buf = map;

	return CUDA_SUCCESS;
}

/**
 * Gdev extension: unmaps device memory from host memory.
 *
 * Parameters:
 * buf - User buffer
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemUnmap(void *buf)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!buf)
		return CUDA_ERROR_INVALID_VALUE;

	handle = ctx->gdev_handle;

	if (gunmap(handle, buf))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

/**
 * Gdev extension: returns physical bus address associated to user buffer.
 * Note that the address is contiguous only within the page boundary.
 *
 * Parameters:
 * addr - Physical bus address obtained
 * p - Pointer to user buffer
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemGetPhysAddr(unsigned long long *addr, void *p)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!addr || !p)
		return CUDA_ERROR_INVALID_VALUE;

	handle = ctx->gdev_handle;

	if (!(*addr = gphysget(handle, p)))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}
