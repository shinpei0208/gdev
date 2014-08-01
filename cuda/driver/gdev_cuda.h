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

#ifndef __GDEV_CUDA_H__
#define __GDEV_CUDA_H__

#define GDEV_CUDA_VERSION 4000

#define GDEV_ARCH_SM_1X 0x50 /* sm_1x */
#define GDEV_ARCH_SM_2X 0xc0 /* sm_2x */
#define GDEV_ARCH_SM_3X 0xe0 /* sm_3x */

#ifndef NULL
#define NULL 0
#endif

#include "gdev_api.h"
#include "gdev_list.h"
#include "gdev_cuda_util.h" /* dependent on libucuda or kcuda. */

#include <linux/version.h>
#ifdef __KERNEL__
#include <linux/sched.h>
#include <linux/spinlock.h>
#if 0
#define GETTID()	task_pid_vnr(current)/*sys_gettid()*/
#else
#define GETTID()	task_pid_nr(current)
#endif
typedef spinlock_t	LOCK_T;
#define LOCK_INIT(l)	spin_lock_init(l)
#define LOCK(l)		spin_lock(l)
#define UNLOCK(l)	spin_unlock(l)
typedef struct timeval	TIME_T;
#define GETTIME(t)	do_gettimeofday(t)
#define YIELD()		yield()
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <pthread.h>
#define GETTID()	syscall(SYS_gettid)
typedef pthread_mutex_t	LOCK_T;
#define LOCK_INIT(l)	pthread_mutex_init(l,NULL)
#define LOCK(l)		pthread_mutex_lock(l)
#define UNLOCK(l)	pthread_mutex_unlock(l)
typedef struct timespec	TIME_T;
#define GETTIME(t)	clock_gettime(CLOCK_MONOTONIC,t);
#define YIELD()		sched_yield()
#endif

struct gdev_cuda_info {
	uint64_t chipset;
	uint64_t mp_count;
	uint64_t warp_count;
	uint64_t warp_size;
};

struct gdev_cuda_param {
	int idx;
	uint32_t offset;
	uint32_t size;
	uint32_t flags;
	struct gdev_cuda_param *next;
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
	uint32_t stack_size;
	uint32_t shared_size;
	uint32_t param_base;
	uint32_t param_size;
	uint32_t param_count;
	struct gdev_cuda_param *param_data;
	uint32_t local_size;
	uint32_t local_size_neg;
};

struct gdev_cuda_fence {
	uint32_t id; /* fence ID returned by the Gdev API. */
	uint64_t addr_ref; /* only used for asynchronous memcpy. */
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
	struct gdev_list event_list;
	struct gdev_cuda_info cuda_info;
	int launch_id;
	int minor;
	unsigned int flags;
	int usage;
	int destroyed;
	pid_t owner;
	pid_t user;
	CUfunc_cache config;
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
	int arch;
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
	int record;
	int complete;
	unsigned int flags;
	TIME_T time;
	struct CUctx_st *ctx;
	struct CUstream_st *stream;
	struct gdev_list list_entry;
};

struct CUstream_st {
	Ghandle gdev_handle;
	struct CUctx_st *ctx;
	struct gdev_list sync_list; /* for gdev_cuda_fence.list_entry */
	struct gdev_list event_list;
	int wait;
};

struct CUgraphicsResource_st {
};

extern int gdev_initialized;
extern int gdev_device_count;
extern struct gdev_list gdev_ctx_list;
extern LOCK_T gdev_ctx_list_lock;

CUresult gdev_cuda_load_cubin(struct CUmod_st *mod, const char *fname);
CUresult gdev_cuda_load_cubin_file(struct CUmod_st *mod, const char *fname);
CUresult gdev_cuda_load_cubin_image(struct CUmod_st *mod, const void *image);
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

static inline uint32_t __gdev_cuda_align_pow2(uint32_t val, uint32_t pow)
{
	if (val == 0)
		return 0;
	if (val & (pow - 1))
		val = (val + pow) & (~(pow - 1));
	return val;
}

static inline uint32_t __gdev_cuda_align_pow2_up(uint32_t val, uint32_t pow)
{
	val = (val == 0) ? 1 : val;
	return __gdev_cuda_align_pow2(val, pow);
}

/* code alignement. */
static inline uint32_t gdev_cuda_align_code_size(uint32_t size)
{
	return __gdev_cuda_align_pow2(size, 0x100);
}

/* constant memory alignement. */
static inline uint32_t gdev_cuda_align_cmem_size(uint32_t size)
{
	return __gdev_cuda_align_pow2(size, 0x100);
}

/* local memory alignement. */
static inline uint32_t gdev_cuda_align_lmem_size(uint32_t size)
{
	return __gdev_cuda_align_pow2(size, 0x10);
}

/* total local memory alignement. */
static inline uint32_t gdev_cuda_align_lmem_size_total(uint32_t size)
{
	return __gdev_cuda_align_pow2(size, 0x20000);
}

/* shared memory alignement. */
static inline uint32_t gdev_cuda_align_smem_size(uint32_t size)
{
	return __gdev_cuda_align_pow2(size, 0x80);
}

/* stack alignement. */
static inline uint32_t gdev_cuda_align_stack_size(uint32_t size)
{
	return __gdev_cuda_align_pow2_up(size, 0x200);
}

/* warp alignement. */
static inline uint32_t gdev_cuda_align_warp_size(uint32_t size)
{
	return __gdev_cuda_align_pow2(size, 0x200);
}

/* memory base alignement. */
static inline uint32_t gdev_cuda_align_base(uint32_t size)
{
	return __gdev_cuda_align_pow2_up(size, 0x1000000);
}

#endif
