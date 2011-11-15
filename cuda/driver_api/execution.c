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
	struct gdev_kernel *k;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
    if (!func)
        return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
    k->block_x = x;
    k->block_y = y;
    k->block_z = z;

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
	struct gdev_kernel *k;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
    if (!func)
        return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	k->smem_size += bytes;

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

/**
 * Sets an integer parameter that will be specified the next time the kernel 
 * corresponding to hfunc will be invoked. offset is a byte offset.
 *
 * Parameters:
 * hfunc - Kernel to add parameter to
 * offset - Offset to add parameter to argument list
 * value - Value of parameter
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
*/
CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	int x;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
    if (!func)
        return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	x = k->cmem_param_segment;
    k->cmem[x].buf[offset/4] = value;
    
	return CUDA_SUCCESS;
}

/**
 * Sets through numbytes the total size in bytes needed by the function 
 * parameters of the kernel corresponding to hfunc.
 *
 * Parameters:
 * hfunc - Kernel to set parameter size for
 * numbytes - Size of parameter list in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	int x;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
    if (!func)
        return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	x = k->cmem_param_segment;
    k->cmem[x].size = numbytes;

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

