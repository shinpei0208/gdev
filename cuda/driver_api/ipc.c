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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

CUresult cuShmAlloc(CUdeviceptr *dptr, int key, unsigned long flags, unsigned int bytesize)
{
	Ghandle handle;
	uint64_t addr;
	uint64_t size = bytesize;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!dptr)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;
	if (!(addr = gmalloc_shm(handle, key, flags, size))) {
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	*dptr = addr;

	return CUDA_SUCCESS;
}

CUresult cuShmFree(CUdeviceptr dptr)
{
	Ghandle handle;
	uint64_t addr = dptr;
	uint64_t size;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	/* wait for all kernels to complete - some may be using the memory. */
	cuCtxSynchronize();

	handle = gdev_ctx_current->gdev_handle;

	if (!(size = gfree_shm(handle, addr)))
		return CUDA_ERROR_INVALID_VALUE;

	return CUDA_SUCCESS;
}
