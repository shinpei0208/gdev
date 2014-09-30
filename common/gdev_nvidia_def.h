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

#ifndef __GDEV_NVIDIA_DEF_H__
#define __GDEV_NVIDIA_DEF_H__

#include "gdev_list.h"
#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif

/**
 * static numbers for nvidia GPUs. 
 */
#define GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT 32 /* by definition? */

/**
 * query values for the device-specific information.
 * they should be >= 0x100.
 */
#define GDEV_NVIDIA_QUERY_MP_COUNT 0x100

/**
 * GPGPU kernel object struct:
 * we use the same kernel struct between user-space and kernel-space.
 *
 * if you want to know how NVIDIA GPGPU kernels work, please reference 
 * the NVIDIA docs and/or the PSCNV wiki available at:
 * https://github.com/pathscale/pscnv/wiki/Nvidia_Compute
 */
struct gdev_kernel {
	uint64_t code_addr; /* code address in VAS */
	uint32_t code_size; /* code size */
	uint32_t code_pc; /* initial program counter */
	struct gdev_cmem {
		uint64_t addr; /* constant memory address in VAS */
		uint32_t size; /* constant memory size */
		uint32_t offset; /* offset in constant memory */
	} cmem[GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT];
	uint32_t cmem_count; /* constant memory count */
	uint32_t param_size; /* kernel parameter size */
	uint32_t *param_buf; /* kernel parameter buffer */
	uint64_t lmem_addr; /* local memory address in VAS */
	uint64_t lmem_size_total; /* local memory size for all threads */
	uint32_t lmem_size; /* local memory size per thread (l[positive]) */
	uint32_t lmem_size_neg; /* data stack size per thread (l[negaive]) */
	uint32_t lmem_base; /* $lbase */
	uint32_t smem_size_func; /* shared memory size specified statically in func bin */
	uint32_t smem_size; /* shared memory size used in real kernel execution */
	uint32_t smem_base; /* $sbase */
	uint32_t warp_stack_size; /* warp stack size (call stack) */
	uint32_t warp_lmem_size; /* total warp memory size */
	uint32_t reg_count; /* register count */
	uint32_t bar_count; /* barrier count */
	uint32_t grid_id; /* grid ID */
	uint32_t grid_x; /* grid dimension X */
	uint32_t grid_y; /* grid dimension Y */
	uint32_t grid_z; /* grid dimension Z */
	uint32_t block_x; /* block dimension X */
	uint32_t block_y; /* block dimension Y */
	uint32_t block_z; /* block dimension Z */
	/* symbol of kernel, for debugging and profiling (Barra back-end) */
	char* name;
};

#endif
