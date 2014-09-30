/*
 * Copyright (C) Shinpei Kato
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
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __GDEV_NVIDIA_H__
#define __GDEV_NVIDIA_H__

#include "gdev_device.h"
#include "gdev_list.h"
#include "gdev_nvidia_def.h"
#include "gdev_system.h"
#include "gdev_time.h"

#define GDEV_NVIDIA_SUBCH_MAX 8

/**
 * subchannel IDs == generic operation IDs. 
 */
#define GDEV_SUBCH_NV_COMPUTE (GDEV_OP_COMPUTE)/* 1 */
#define GDEV_SUBCH_NV_M2MF (GDEV_OP_MEMCPY)/* 2 */
#define GDEV_SUBCH_NV_PCOPY0 (GDEV_OP_MEMCPY_ASYNC) /* 3 */
#define GDEV_SUBCH_NV_PCOPY1 (GDEV_SUBCH_NV_PCOPY0 + 1) /* 4 */
#define GDEV_SUBCH_NV_PCOPY2 (GDEV_SUBCH_NV_PCOPY0 + 2) /* 5 */

#define GDEV_FENCE_BUF_SIZE 0x10000 /* 64KB */
#define GDEV_FENCE_QUERY_SIZE 0x10 /* aligned with nvc0's query */
#define GDEV_FENCE_COUNT (GDEV_FENCE_BUF_SIZE / GDEV_FENCE_QUERY_SIZE)

/**
 * map host and device memory if the allocated size is small.
 * it will help to reduce the cost of memcpy.
 */
#define GDEV_MEM_MAPPABLE_LIMIT 0x400000 /* small page size 4MB */

/**
 * virutal address space available for user buffers.
 */
#define GDEV_VAS_USER_START 0x20000000
#define GDEV_VAS_USER_END (1ull << 40)
#define GDEV_VAS_SIZE GDEV_VAS_USER_END

/**
 * memory types.
 */
#define GDEV_MEM_DEVICE 0x01
#define GDEV_MEM_DMA 0x02

/**
 * some stupid distros have "memcpy" macros. we don't need it.
 */
#ifdef memcpy
#undef memcpy
#endif

/**
 * Gdev shared memory information:
 */
struct gdev_shm {
	struct gdev_mem *holder; /* current memory holder */
	struct gdev_list mem_list; /* list of memory objects attached */
	struct gdev_list list_entry; /* entry to the list of shared memory */
	gdev_mutex_t mutex;
	uint64_t size;
	int prio; /* highest prio among users (effective only for master) */
	int users; /* number of users (effective only for master) */
	int key; /* key value of this shared memory */
	int id; /* indentifier of this shared memory */
	int implicit; /* true if created due to a lack of memory space. */
	void *bo; /* private buffer object */
};

/**
 * virtual address space (VAS) object struct:
 *
 * NVIDIA GPUs support virtual memory (VM) with 40 bits addressing.
 * VAS hence ranges in [0:1<<40]. In particular, the pscnv bo function will
 * allocate [0x20000000:1<<40] to any buffers in so called global memory,
 * local memory, and constant memory. the rest of VAS is used for different 
 * purposes, e.g., for shared memory.
 * CUDA programs access these memory spaces as follows:
 * g[$reg] redirects to one of g[$reg], l[$reg-$lbase], and s[$reg-$sbase],
 * depending on how local memory and shared memory are set up.
 * in other words, g[$reg] may reference global memory, local memory, and
 * shared memory.
 * $lbase and $sbase are respectively local memory and shared memory base
 * addresses, which are configured when GPU kernels are launched.
 * l[0] and g[$lbase] reference the same address, so do s[0] and g[$sbase].
 * constant memory, c[], is another type of memory space that is often used
 * to store GPU kernels' parameters (arguments).
 * global memory, local memory, and constant memory are usually mapped on
 * device memory, a.k.a., video RAM (VRAM), though they could also be mapped
 * on host memory, a.k.a., system RAM (SysRAM), while shared memory is always
 * mapped on SRAM present in each MP.
 */
struct gdev_vas {
	int vid; /* vritual address space ID. */
	void *handle; /* Gdev API handle. */
	void *pvas; /* driver private object. */
	struct gdev_device *gdev; /* vas is associated with a specific device. */
	struct gdev_list mem_list; /* list of device memory spaces. */
	struct gdev_list dma_mem_list; /* list of host dma memory spaces. */
	struct gdev_list list_entry; /* entry to the vas list. */
	gdev_lock_t lock;
	int prio;
};

/**
 * GPU context object struct:
 */
struct gdev_ctx {
	int cid; /* context ID. */
	void *pctx; /* driver private object. */
	struct gdev_vas *vas; /* chan is associated with a specific vas object. */
	struct gdev_fifo {
		volatile uint32_t *regs; /* channel control registers. */
		void *ib_bo; /* driver private object. */
		uint32_t *ib_map;
		uint32_t ib_order;
		uint64_t ib_base;
		uint32_t ib_mask;
		uint32_t ib_put;
		uint32_t ib_get;
		void *pb_bo; /* driver private object. */
		uint32_t *pb_map;
		uint32_t pb_order;
		uint64_t pb_base;
		uint32_t pb_mask;
		uint32_t pb_size;
		uint32_t pb_pos;
		uint32_t pb_put;
		uint32_t pb_get;
		void (*space)(struct gdev_ctx *, uint32_t);
		void (*push)(struct gdev_ctx *, uint64_t, uint32_t, int);
		void (*kick)(struct gdev_ctx *);
		void (*update_get)(struct gdev_ctx *);
	} fifo; /* command FIFO queue struct. */
	struct gdev_fence { /* fence objects (for compute and dma). */
		void *bo; /* driver private object. */
		uint32_t *map;
		uint64_t addr;
		uint32_t seq;
	} fence;
	struct gdev_intr { /* notifier objects (for compute and dma). */
		void *bo; /* driver private object. */
		uint64_t addr;
	} notify;
	uint32_t dummy;
	void *pdata; /* arch-specific private data object. */
	struct gdev_desc {
	    void *bo;
	    uint32_t *map;
	    uint32_t addr;
	} desc; /* compute desc struct. But really, need to structure for each launch grid */
};

/**
 * device/host memory object struct:
 */
struct gdev_mem {
	void *bo; /* driver private object */
	struct gdev_vas *vas; /* mem is associated with a specific vas object */
	struct gdev_list list_entry_heap; /* entry to heap list */
	struct gdev_list list_entry_shm; /* entry to shared memory list */
	struct gdev_shm *shm; /* shared memory information */
	struct gdev_mem *swap_mem; /* device memory for temporal swap */
	void *swap_buf; /* host buffer for swap */
	int evicted; /* 1 if evicted, 0 otherwise */
	uint64_t size; /* memory size */
	uint64_t addr; /* virtual memory address */
	int type; /* device or host dma? */
	void *map; /* memory-mapped buffer */
	int map_users; /* # of users referencing the map */
	void *pdata; /* arch-specific private data object. */
};

/**
 * private compute functions. 
 */
struct gdev_compute {
	int (*launch)(struct gdev_ctx *, struct gdev_kernel *);
	uint32_t (*fence_read)(struct gdev_ctx *, uint32_t);
	void (*fence_write)(struct gdev_ctx *, int, uint32_t);
	void (*fence_reset)(struct gdev_ctx *, uint32_t);
	void (*memcpy)(struct gdev_ctx *, uint64_t, uint64_t, uint32_t);
	void (*memcpy_async)(struct gdev_ctx *, uint64_t, uint64_t, uint32_t);
	void (*membar)(struct gdev_ctx *);
	void (*notify_intr)(struct gdev_ctx *);
	void (*init)(struct gdev_ctx *);
};

/**
 * generic memory list functions for NVIDIA implementations.
 */
void gdev_nvidia_mem_setup(struct gdev_mem *mem, struct gdev_vas *vas, int type);
void gdev_nvidia_mem_list_add(struct gdev_mem *mem);
void gdev_nvidia_mem_list_del(struct gdev_mem *mem);

/**
 * chipset specific functions.
 */
void nvc0_compute_setup(struct gdev_device *gdev);
void nve4_compute_setup(struct gdev_device *gdev);
void barra_compute_setup(struct gdev_device *gdev);

/**
 * OS driver and user-space runtime depen functions.
 */
int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *result);
struct gdev_device *gdev_raw_dev_open(int minor);
void gdev_raw_dev_close(struct gdev_device *gdev);
struct gdev_vas *gdev_raw_vas_new(struct gdev_device *gdev, uint64_t size);
void gdev_raw_vas_free(struct gdev_vas *vas);
struct gdev_ctx *gdev_raw_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas);
void gdev_raw_ctx_free(struct gdev_ctx *ctx);
struct gdev_mem *gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t size);
struct gdev_mem *gdev_raw_mem_alloc_dma(struct gdev_vas *vas, uint64_t size);
void gdev_raw_mem_free(struct gdev_mem *mem);
struct gdev_mem *gdev_raw_swap_alloc(struct gdev_device *gdev, uint64_t size);
void gdev_raw_swap_free(struct gdev_mem *mem);
struct gdev_mem *gdev_raw_mem_share(struct gdev_vas *vas, struct gdev_mem *mem);
void gdev_raw_mem_unshare(struct gdev_mem *mem);
void *gdev_raw_mem_map(struct gdev_mem *mem);
void gdev_raw_mem_unmap(struct gdev_mem *mem, void *map);
uint64_t gdev_raw_mem_phys_getaddr(struct gdev_mem *mem, uint64_t offset);
uint32_t gdev_raw_read32(struct gdev_mem *mem, uint64_t addr);
void gdev_raw_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val);
int gdev_raw_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size);
int gdev_raw_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size);

static inline void __gdev_fire_ring(struct gdev_ctx *ctx)
{
	if (ctx->fifo.pb_pos != ctx->fifo.pb_put) {
		uint64_t base = ctx->fifo.pb_base + ctx->fifo.pb_put;
		uint32_t len;
		if (ctx->fifo.pb_pos > ctx->fifo.pb_put) {
			len = ctx->fifo.pb_pos - ctx->fifo.pb_put;
		} else {
			len = ctx->fifo.pb_size - ctx->fifo.pb_put;
			ctx->fifo.push(ctx, base, len, 0);
			ctx->fifo.pb_put = 0;
			base = ctx->fifo.pb_base;
			len = ctx->fifo.pb_pos;
		}
		if (len > 0)
			ctx->fifo.push(ctx, base, len, 0);
		ctx->fifo.pb_put = ctx->fifo.pb_pos;
		if (ctx->fifo.kick)
			ctx->fifo.kick(ctx);
	}
}

static inline void __gdev_out_ring(struct gdev_ctx *ctx, uint32_t word)
{
	while (((ctx->fifo.pb_pos + 4) & ctx->fifo.pb_mask) == ctx->fifo.pb_get) {
		uint32_t old = ctx->fifo.pb_get;
		//__gdev_fire_ring(ctx);
		ctx->fifo.update_get(ctx);
		if (old == ctx->fifo.pb_get) {
			SCHED_YIELD();
		}
	}
	ctx->fifo.pb_map[ctx->fifo.pb_pos/4] = word;
	ctx->fifo.pb_pos += 4;
	ctx->fifo.pb_pos &= ctx->fifo.pb_mask;
}

static inline void __gdev_ring_space(struct gdev_ctx *ctx, uint32_t word)
{
	if (ctx->fifo.space)
		ctx->fifo.space(ctx, word);
}

static inline void __gdev_begin_ring_nv50(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, mthd | (subc<<13) | (len<<18));
}

static inline void __gdev_begin_ring_nv50_const(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, mthd | (subc<<13) | (len<<18) | (0x4<<28));
}

static inline void __gdev_begin_ring_nvc0(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, (0x2<<28) | (len<<16) | (subc<<13) | (mthd>>2));
}

static inline void __gdev_begin_ring_nvc0_const(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, (0x6<<28) | (len<<16) | (subc<<13) | (mthd>>2));
}


static inline void __gdev_begin_ring_nve4(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, (0x2<<28) | (len<<16) | (subc<<13) | (mthd>>2));
}

static inline void __gdev_begin_ring_nve4_const(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, (0x6<<28) | (len<<16) | (subc<<13) | (mthd>>2));
}

static inline void __gdev_begin_ring_nve4_il(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, (0x8<<28) | (len<<16) | (subc<<13) | (mthd>>2));
}
static inline void __gdev_begin_ring_nve4_1l(struct gdev_ctx *ctx, int subc, int mthd, int len)
{
	__gdev_out_ring(ctx, (0xa<<28) | (len<<16) | (subc<<13) | (mthd>>2));
}


#endif
