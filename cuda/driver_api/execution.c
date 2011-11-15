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

CUresult cuFuncGetAttribute
(int *pi, CUfunction_attribute attrib, CUfunction hfunc) 
{
	GDEV_PRINT("cuFuncGetAttribute: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Specifies the x, y, and z dimensions of the thread blocks that are created 
 * when the kernel given by hfunc is launched.
 *
 * Parameters:
 * hfunc - Kernel to specify dimensions of
 * x - X dimension
 * y - Y dimension
 * z - Z dimension
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE (what is this?), 
 * CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *ctx = mod->ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
    if (!func)
        return CUDA_ERROR_INVALID_VALUE;

    func->kernel.block_x = x;
    func->kernel.block_y = y;
    func->kernel.block_z = z;

	return CUDA_SUCCESS;
}

/**
 * Sets through bytes the amount of dynamic shared memory that will be 
 * available to each thread block when the kernel given by hfunc is launched.
 *
 * Parameters:
 * hfunc - Kernel to specify dynamic shared-memory size for
 * bytes - Dynamic shared-memory size per thread in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE (what is this?), 
 * CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
{
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *ctx = mod->ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
    if (!func)
        return CUDA_ERROR_INVALID_VALUE;

	func->kernel.smem_size += bytes;

	return CUDA_SUCCESS;
}

CUresult cuLaunch(CUfunction f)
{
	return CUDA_SUCCESS;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
	return CUDA_SUCCESS;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
	GDEV_PRINT("cuLaunchGridAsync: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
{
	GDEV_PRINT("cuParamSetf: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
	return CUDA_SUCCESS;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
	return CUDA_SUCCESS;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
	GDEV_PRINT("cuParamSetTexRef: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
	return CUDA_SUCCESS;
}

