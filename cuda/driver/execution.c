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
#include "gdev_api.h"
#include "gdev_autogen.h"
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
	CUresult res;
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	int nr_max_threads = ctx->cuda_info.warp_size * ctx->cuda_info.warp_count;
	
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
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
	CUresult res;
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	k->smem_size = gdev_cuda_align_smem_size(k->smem_size_func + bytes);

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
	CUresult res;
	struct CUfunc_st *func = f;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	struct gdev_cuda_fence *fence;
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func || grid_width <= 0 || grid_height <= 0)
		return CUDA_ERROR_INVALID_VALUE;
	if (!(fence = (struct gdev_cuda_fence *)MALLOC(sizeof(*fence))))
		return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;

	k = &func->kernel;
	k->grid_x = grid_width;
	k->grid_y = grid_height;
	k->grid_z = 1;
	k->grid_id = ++ctx->launch_id;
	k->name = func->raw_func.name;


#ifdef GDEV_DRIVER_NOUVEAU /* this is a quick hack until Nouveau supports flexible vspace */
	k->smem_base = 0xe << 24; 
	k->lmem_base = 0xf << 24;
#else
	k->smem_base = gdev_cuda_align_base(0);
	k->lmem_base = k->smem_base + gdev_cuda_align_base(k->smem_size);
#endif

	handle = cur->gdev_handle;

	if (glaunch(handle, k, &fence->id))
		return CUDA_ERROR_LAUNCH_FAILED;
	fence->addr_ref = 0; /* no address to unreference later. */
	gdev_list_init(&fence->list_entry, fence);
	gdev_list_add(&fence->list_entry, &ctx->sync_list);

	return CUDA_SUCCESS;
}

CUresult cuLaunchGridAsync
(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
	GDEV_PRINT("cuLaunchGridAsync: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Invokes the kernel @f on a @gridDimX x @gridDimY x @gridDimZ grid of blocks. 
 * Each block contains @blockDimX x @blockDimY x @blockDimZ threads.
 * @sharedMemBytes sets the amount of dynamic shared memory that will be 
 * available to each thread block.
 *
 * cuLaunchKernel() can optionally be associated to a stream by passing a 
 * non-zero hStream argument.
 *
 * Kernel parameters to @f can be specified in one of two ways:
 *
 * 1) Kernel parameters can be specified via kernelParams. If f has N 
 * parameters, then kernelParams needs to be an array of N pointers. Each of 
 * kernelParams[0] through kernelParams[N-1] must point to a region of memory 
 * from which the actual kernel parameter will be copied. The number of kernel 
 * parameters and their offsets and sizes do not need to be specified as that 
 * information is retrieved directly from the kernel's image.
 *
 * 2) Kernel parameters can also be packaged by the application into a single 
 * buffer that is passed in via the extra parameter. This places the burden on
 * the application of knowing each kernel parameter's size and alignment/
 * padding within the buffer. Here is an example of using the extra parameter 
 * in this manner:
 *
 *  size_t argBufferSize;
 *  char argBuffer[256];
 *
 *  // populate argBuffer and argBufferSize
 *
 *  void *config[] = {
 *      CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
 *      CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
 *      CU_LAUNCH_PARAM_END
 *  };
 *  status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);
 *
 * The extra parameter exists to allow cuLaunchKernel to take additional less 
 * commonly used arguments. extra specifies a list of names of extra settings 
 * and their corresponding values. Each extra setting name is immediately 
 * followed by the corresponding value. The list must be terminated with 
 * either NULL or CU_LAUNCH_PARAM_END.
 *
 *  CU_LAUNCH_PARAM_END, which indicates the end of the extra array;
 *  CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next value in 
 *  extra will be a pointer to a buffer containing all the kernel parameters 
 *  for launching kernel f;
 *  CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next value in extra
 *  will be a pointer to a size_t containing the size of the buffer specified 
 *  with CU_LAUNCH_PARAM_BUFFER_POINTER;
 *
 * The error CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters 
 * are specified with both kernelParams and extra (i.e. both kernelParams and 
 * extra are non-NULL).
 *
 * Calling cuLaunchKernel() sets persistent function state that is the same as 
 * function state set through the following deprecated APIs:
 *
 * cuFuncSetBlockShape() cuFuncSetSharedSize() cuParamSetSize() cuParamSeti() 
 * cuParamSetf() cuParamSetv()
 *
 * When the kernel @f is launched via cuLaunchKernel(), the previous block 
 * shape, shared size and parameter info associated with @f is overwritten.
 *
 * Note that to use cuLaunchKernel(), the kernel @f must either have been 
 * compiled with toolchain version 3.2 or later so that it will contain kernel 
 * parameter information, or have no kernel parameters. If either of these 
 * conditions is not met, then cuLaunchKernel() will return 
 * CUDA_ERROR_INVALID_IMAGE.
 *
 * Parameters:
 * f - Kernel to launch
 * gridDimX	- Width of grid in blocks
 * gridDimY - Height of grid in blocks
 * gridDimZ - Depth of grid in blocks
 * blockDimX - X dimension of each thread block
 * blockDimY - Y dimension of each thread block
 * blockDimZ - Z dimension of each thread block
 * sharedMemBytes - Dynamic shared-memory size per thread block in bytes
 * hStream - Stream identifier
 * kernelParams - Array of pointers to kernel parameters
 * extra - Extra options
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, 
 * CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_FAILED,
 * CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, CUDA_ERROR_LAUNCH_TIMEOUT, 
 * CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, 
 * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED 
 */
CUresult cuLaunchKernel
(CUfunction f, 
 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
 unsigned int sharedMemBytes, CUstream hStream, 
 void **kernelParams, void **extra)
{
	struct gdev_cuda_raw_func *rf;
	struct gdev_cuda_param *param_data;
	CUresult res;

	if (hStream) {
		GDEV_PRINT("cuLaunchKernel: Stream is not supported.\n");
		return CUDA_ERROR_INVALID_HANDLE;
	}

	if (extra) {
		GDEV_PRINT("cuLaunchKernel: Extra Parameters are not supported.\n");
		return CUDA_ERROR_INVALID_HANDLE;
	}

	res = cuFuncSetSharedSize(f, sharedMemBytes);
	if (res != CUDA_SUCCESS)
		return res;

	res = cuFuncSetBlockShape(f, blockDimX, blockDimY, blockDimZ);
	if (res != CUDA_SUCCESS)
		return res;

	rf = &f->raw_func;
	param_data = rf->param_data;
	while (param_data) {
		void *p = kernelParams[param_data->idx];
		int offset = param_data->offset;
		uint32_t size = param_data->size;
		cuParamSetv(f, offset, p, size);
		param_data = param_data->next;
	}

	res = cuParamSetSize(f, rf->param_size);
	if (res != CUDA_SUCCESS)
		return res;

	res = cuLaunchGrid(f, gridDimX, gridDimY);
	if (res != CUDA_SUCCESS)
		return res;

	return CUDA_SUCCESS;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
{
	CUresult res;
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	f = &func->raw_func;
	((float *)k->param_buf)[(f->param_base + offset) / 4] = value;
	
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
	CUresult res;
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
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
	CUresult res;
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
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
	CUresult res;
	struct CUfunc_st *func = hfunc;
	struct CUmod_st *mod = func->mod;
	struct CUctx_st *cur;
	struct CUctx_st *ctx = mod->ctx;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&cur);
	if (res != CUDA_SUCCESS)
		return res;

	if (!ctx || ctx != cur)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!func)
		return CUDA_ERROR_INVALID_VALUE;

	k = &func->kernel;
	f = &func->raw_func;
	memcpy(&k->param_buf[(f->param_base + offset) / 4], ptr, numbytes);
	
	return CUDA_SUCCESS;
}

