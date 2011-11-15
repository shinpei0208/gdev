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

/***************************************************************************
 * Currently, a very limited set of memory management functions is supported
 * under Gdev. There are lots of things to be additionally implemented...
 ***************************************************************************/

/**
 * Allocates bytesize bytes of linear memory on the device and returns in 
 * @dptr a pointer to the allocated memory. The allocated memory is suitably 
 * aligned for any kind of variable. The memory is not cleared. If bytesize 
 * is 0, cuMemAlloc() returns CUDA_ERROR_INVALID_VALUE.
 *
 * Parameters:
 * dptr - Returned device pointer
 * bytesize - Requested allocation size in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_OUT_OF_MEMORY 
 */
CUresult cuMemAlloc(CUdeviceptr *dptr, unsigned int bytesize)
{
	gdev_handle_t *handle;
	uint64_t addr;
	uint32_t size = bytesize;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!dptr)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;
	if (!(addr = gmalloc(handle, size)))
		return CUDA_ERROR_OUT_OF_MEMORY;

	*dptr = addr;

	return CUDA_SUCCESS;
}

/**
 * Frees the memory space pointed to by dptr, which must have been returned 
 * by a previous call to cuMemAlloc() or cuMemAllocPitch().
 *
 * Parameters:
 * dptr 	- Pointer to memory to free
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemFree(CUdeviceptr dptr)
{
	gdev_handle_t *handle;
	uint64_t addr = dptr;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	handle = gdev_ctx_current->gdev_handle;

	if (gfree(handle, addr))
		return CUDA_ERROR_INVALID_VALUE;

	return CUDA_SUCCESS;
}

CUresult cuMemAllocHost(void **pp, unsigned int bytesize)
{
	return CUDA_SUCCESS;
}

CUresult cuMemFreeHost (void *p)
{
	return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}
