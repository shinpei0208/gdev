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
	int nr_max_threads = ctx->cuda_info.warp_size * 32;
	
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func || x <= 0 || y <= 0 || z <= 0 || x * y * z > nr_max_threads)
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
	k->smem_size += gdev_cuda_align_smem_size(bytes);

	return CUDA_SUCCESS;
}

/**
 * Invokes the kernel f on a 1 x 1 x 1 grid of blocks. The block contains the
 *  number of threads specified by a previous call to cuFuncSetBlockShape().
 *
 * Parameters:
 * f 	- Kernel to launch
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, 
 * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING 
 */
CUresult cuLaunch(CUfunction f)
{
	return cuLaunchGrid(f, 1, 1);
}

/**
 * Invokes the kernel f on a grid_width x grid_height grid of blocks. Each 
 * block contains the number of threads specified by a previous call to 
 * cuFuncSetBlockShape().
 *
 * Parameters:
 * f - Kernel to launch
 * grid_width - Width of grid in blocks
 * grid_height - Height of grid in blocks
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, 
 * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING 
 */
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
	struct CUfunc_st *func = f;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	Ghandle handle;
	uint32_t id;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func || grid_width <= 0 || grid_height <= 0)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	k->grid_x = grid_width;
	k->grid_y = grid_height;
	k->grid_z = 1;
	k->grid_id = 1;

	k->smem_base = gdev_cuda_align_base(ctx->data_size);
	k->lmem_base = gdev_cuda_align_base(k->smem_base + k->smem_size);

	handle = gdev_ctx_current->gdev_handle;

	if (glaunch(handle, k, &id))
		return CUDA_ERROR_LAUNCH_FAILED;

	/* if timeout is required, specify gdev_time value instead of NULL. 
	   this sync should be moved to cuCtxSynchronize(). */
	if (gsync(handle, id, NULL))
		return CUDA_ERROR_LAUNCH_TIMEOUT;

	return CUDA_SUCCESS;
}

CUresult cuLaunchGridAsync
(CUfunction f, int grid_width, int grid_height, CUstream hStream)
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
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	f = &func->raw_func;
	k->param_buf[(f->param_base + offset) / 4] = value;
	
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
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	f = &func->raw_func;
	if (k->param_size - f->param_base < numbytes)
		return CUDA_ERROR_INVALID_VALUE;
	else
		k->param_size = f->param_base + numbytes;

	return CUDA_SUCCESS;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
	GDEV_PRINT("cuParamSetTexRef: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuParamSetv
(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx || ctx != gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	f = &func->raw_func;
	memcpy(&k->param_buf[(f->param_base + offset) / 4], ptr, numbytes);
	
	return CUDA_SUCCESS;
}

