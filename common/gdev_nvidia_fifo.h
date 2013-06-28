/*
 * Copyright (C) Shinpei Kato
 *
 * Nagoya University
 * Parallel and Distributed Systems Lab.
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

#ifndef __GDEV_NVIDIA_FIFO_H__
#define __GDEV_NVIDIA_FIFO_H__

#include "gdev_nvidia.h"

void gdev_fifo_push(struct gdev_ctx *ctx, uint64_t base, uint32_t len, int flags);
void gdev_fifo_update_get(struct gdev_ctx *ctx);

static inline uint32_t __gdev_fifo_read_reg(struct gdev_ctx *ctx, uint32_t reg)
{
	/* don't forget (unsigned long) cast... */
	return IOREAD32((unsigned long)ctx->fifo.regs + reg);
}

static inline void __gdev_fifo_write_reg(struct gdev_ctx *ctx, uint32_t reg, uint32_t val)
{
	/* don't forget (unsigned long) cast... */
	IOWRITE32(val, (unsigned long)ctx->fifo.regs + reg);
}


#endif
