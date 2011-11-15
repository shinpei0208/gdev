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
 * Takes a filename fname and loads the corresponding module module into the
 * current context. The CUDA driver API does not attempt to lazily allocate 
 * the resources needed by a module; if the memory for functions and data 
 * (constant and global) needed by the module cannot be allocated, 
 * cuModuleLoad() fails. The file should be a cubin file as output by nvcc 
 * or a PTX file, either as output by nvcc or handwrtten.
 *
 * Parameters:
 * module - Returned module
 * fname - Filename of module to load
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND, 
 * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_FILE_NOT_FOUND 
 */
CUresult cuModuleLoad(CUmodule *module, const char *fname)
{
	CUresult res;
	struct CUmod_st *mod;
	struct CUctx_st *ctx;
	gdev_handle_t *handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!module || fname)
		return CUDA_ERROR_INVALID_VALUE;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	ctx = gdev_ctx_current;
	handle = ctx->gdev_handle;

	if (!(mod = malloc(sizeof(*mod)))) {
		res = CUDA_ERROR_OUT_OF_MEMORY;
		goto fail_malloc_mod;
	}

	/* load the cubin image from the given object file. */
	if ((res = gdev_cuda_load_cubin(mod, fname)) != CUDA_SUCCESS)
		goto fail_load_cubin;

	/* setup the kernels based on the cubin data. */
	gdev_cuda_setup_kernels(mod, &ctx->cuda_info);

	/* allocate local memory, and assign it to each function. */
	if (!(mod->local_addr = gmalloc(handle, mod->local_size))) {
		res = CUDA_ERROR_OUT_OF_MEMORY;
		goto fail_gmalloc_local;
	}

	if ((res = gdev_cuda_assign_local(mod)))
		goto fail_assign_local;

	/* allocate code and constant memory and assign it to each function. */
	if (!(mod->image_addr = gmalloc(handle, mod->image_size)))
		goto fail_gmalloc_image;
	/* this malloc() and memcpy() in gdev_cuda_setup_image() could be
	   removed if we use gmalloc_host() here, the following is just an easy
	   implementation, and doesn't affect performance much anyway. */
	if (!(mod->image_buf = malloc(mod->image_size))) {
		res = CUDA_ERROR_OUT_OF_MEMORY;
		goto fail_malloc_image;
	}
	memset(mod->image_buf, 0, mod->image_size);
	if ((res = gdev_cuda_assign_image(mod)))
		goto fail_assign_image;

	/* transfer the code and constant memory onto the device. */
	if (gmemcpy_to_device(handle, mod->image_addr, mod->image_buf, 
						  mod->image_size)) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_gmemcpy;
	}

	*module = mod;

	return CUDA_SUCCESS;

fail_gmemcpy:
fail_assign_image:
	free(mod->image_buf);
fail_malloc_image:
	gfree(handle, mod->image_addr);
fail_gmalloc_image:
fail_assign_local:
	gfree(handle, mod->local_addr);
fail_gmalloc_local:
	gdev_cuda_unload_cubin(mod);
fail_load_cubin:
	free(mod);
fail_malloc_mod:
	*module = NULL;
	return res;
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
	printf("cuModuleLoadFatBinary: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Unloads a module hmod from the current context.
 *
 * Parameters:
 * hmod - Module to unload
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuModuleUnload(CUmodule hmod)
{
	CUresult res;
	struct CUmod_st *mod = hmod;
	gdev_handle_t *handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!mod)
		return CUDA_ERROR_INVALID_VALUE;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	handle = gdev_ctx_current->gdev_handle;

	free(mod->image_buf);
	gfree(handle, mod->image_addr);
	gfree(handle, mod->local_addr);

	if ((res = gdev_cuda_unload_cubin(mod)) != CUDA_SUCCESS)
		return res;

	free(mod);

	return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
	return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
	printf("cuModuleLoadData: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
	printf("cuModuleLoadDataEx: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
	printf("cuModuleGetGlobal: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
	printf("cuModuleGetTexRef: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}
