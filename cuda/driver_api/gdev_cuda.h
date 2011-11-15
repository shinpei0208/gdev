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

#ifndef __GDEV_CUDA_H__
#define __GDEV_CUDA_H__

#define GDEV_CUDA_VERSION 4000
#define GDEV_CUDA_USER_PARAM_BASE 0x20
#define GDEV_CUDA_CMEM_SEGMENT_COUNT 16 /* by definition? */

#ifndef NULL
#define NULL 0
#endif

#include "gdev_api.h"
#include "gdev_list.h"

struct gdev_cuda_info {
	uint32_t mp_count;
	uint32_t warp_count;
	uint32_t warp_size;
};

struct gdev_cuda_raw_func {
	char *name;
	void *code_buf;
	uint32_t code_size;
	struct gdev_cuda_cmem {
		void *buf;
		uint32_t size;
	} cmem[GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT];
	uint32_t reg_count;
	uint32_t bar_count;
	uint32_t stack_depth;
	uint32_t shared_size;
	uint32_t param_size;
	uint32_t local_size;
	uint32_t local_size_neg;
};

struct CUctx_st {
	gdev_handle_t *gdev_handle;
	gdev_list_t list_entry;
	struct gdev_cuda_info cuda_info;
};

struct CUmod_st {
	FILE *fp;
	void *bin;
	void *image_buf;
	uint64_t image_addr;
	uint64_t local_addr;
	uint32_t image_size;
	uint32_t local_size;
	uint32_t func_count;
	gdev_list_t func_list;
};

struct CUfunc_st {
	struct gdev_kernel kernel;
	struct gdev_cuda_raw_func raw_func;
	gdev_list_t list_entry;
};

struct CUtexref_st {
};

CUresult gdev_cuda_load_cubin(struct CUmod_st *mod, const char *fname);
CUresult gdev_cuda_unload_cubin(struct CUmod_st *mod);
void gdev_cuda_setup_kernels(struct CUmod_st *mod, struct gdev_cuda_info *info);
CUresult gdev_cuda_assign_image(struct CUmod_st *mod);
CUresult gdev_cuda_assign_local(struct CUmod_st *mod);

extern int gdev_initialized;
extern int gdev_device_count;
struct CUctx_st *gdev_ctx_current;
extern gdev_list_t gdev_ctx_list;

#endif
