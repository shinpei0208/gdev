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

#ifndef NULL
#define NULL 0
#endif

#ifdef __KERNEL__
#include "gdev_drv.h"
#else
#include "gdev_lib.h"
#endif
#include "gdev_api.h"
#include "gdev_list.h"
#include "gdev_cuda_util.h" /* dependent on libcuda or kcuda. */

struct gdev_cuda_info {
	uint64_t chipset;
	uint64_t mp_count;
	uint64_t warp_count;
	uint64_t warp_size;
};

struct gdev_cuda_raw_func {
	char *name;
	void *code_buf;
	uint32_t code_size;
	struct {
		void *buf;
		uint32_t size;
	} cmem[GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT]; /* local to functions. */
	uint32_t reg_count;
	uint32_t bar_count;
	uint32_t stack_depth;
	uint32_t shared_size;
	uint32_t param_base;
	uint32_t param_size;
	uint32_t param_count;
	struct {
		uint32_t offset;
		uint32_t size;
		uint32_t flags;
	} *param_info;
	uint32_t local_size;
	uint32_t local_size_neg;
};

struct gdev_cuda_launch {
	uint32_t id; /* kernel ID returned by the launch function. */
	struct gdev_list list_entry; /* entry to synchronization list. */
};

struct gdev_cuda_const_symbol {
	int idx; /* cX[] index. */
	char *name;
	uint32_t offset; /* offset in cX[]. */
	uint32_t size; /* size of const value. */
	struct gdev_list list_entry; /* entry to symbol list. */
};

struct CUctx_st {
	Ghandle gdev_handle;
	struct gdev_list list_entry; /* entry to ctx_list. */
	struct gdev_list sync_list;
	struct gdev_cuda_info cuda_info;
	uint64_t data_size;
	int launch_id;
};

struct CUmod_st {
	file_t *fp;
	void *bin;
	uint64_t code_addr;
	uint32_t code_size;
	uint64_t sdata_addr;
	uint32_t sdata_size;
	struct {
		uint64_t addr;
		uint32_t size;
		uint32_t raw_size;
		void *buf;
	} cmem[GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT]; /* global to functions. */
	uint32_t func_count;
	uint32_t symbol_count;
	struct gdev_list func_list;
	struct gdev_list symbol_list;
	struct CUctx_st *ctx;
};

struct CUfunc_st {
	struct gdev_kernel kernel;
	struct gdev_cuda_raw_func raw_func;
	struct gdev_list list_entry;
	struct CUmod_st *mod;
};

struct CUtexref_st {
};

struct CUsurfref_st {
};

struct CUevent_st {
};

struct CUstream_st {
};

struct CUgraphicsResource_st {
};

extern int gdev_initialized;
extern int gdev_device_count;
extern struct CUctx_st *gdev_ctx_current;
extern struct gdev_list gdev_ctx_list;

CUresult gdev_cuda_load_cubin(struct CUmod_st *mod, const char *fname);
CUresult gdev_cuda_unload_cubin(struct CUmod_st *mod);
CUresult gdev_cuda_construct_kernels
(struct CUmod_st *mod, struct gdev_cuda_info *cuda_info);
CUresult gdev_cuda_destruct_kernels(struct CUmod_st *mod);
CUresult gdev_cuda_locate_sdata(struct CUmod_st *mod);
CUresult gdev_cuda_locate_code(struct CUmod_st *mod);
CUresult gdev_cuda_memcpy_code(struct CUmod_st *mod, void *buf);
CUresult gdev_cuda_search_function
(struct CUfunc_st **pptr, struct CUmod_st *mod, const char *name);
CUresult gdev_cuda_search_symbol
(uint64_t *addr, uint32_t *size, struct CUmod_st *mod, const char *name);

/* code alignement. */
static inline uint32_t gdev_cuda_align_code_size(uint32_t size)
{
	if (size & 0xff)
		size = (size + 0x100) & ~0xff;
	return size;
}

/* constant memory alignement. */
static inline uint32_t gdev_cuda_align_cmem_size(uint32_t size)
{
	if (size & 0xff)
		size = (size + 0x100) & ~0xff;
	return size;
}

/* local memory alignement. */
static inline uint32_t gdev_cuda_align_lmem_size(uint32_t size)
{
	if (size & 0xf)
		size = (size + 0x10) & ~0xf;
	return size;
}

/* total local memory alignement. */
static inline uint32_t gdev_cuda_align_lmem_size_total(uint32_t size)
{
	if (size & 0x1fffff)
		size = (size + 0x200000) & ~0x1fffff;
	return size;
}

/* shared memory alignement. */
static inline uint32_t gdev_cuda_align_smem_size(uint32_t size)
{
	if (size & 0x7f)
		size = (size + 0x80) & (~0x7f);
	return size;
}

/* warp alignement. */
static inline uint32_t gdev_cuda_align_warp_size(uint32_t size)
{
	if (size & 0x7ff)
		size = (size + 0x800) & (~0x7ff);
	return size;
}

/* memory base alignement. */
static inline uint32_t gdev_cuda_align_base(uint32_t size)
{
	if (size & 0xffffff)
		size = (size + 0x1000000) & (~0xffffff);
	return size;
}

#endif
