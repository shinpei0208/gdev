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

#include "gdev_nvidia_fifo.h"

void gdev_fifo_push(struct gdev_ctx *ctx, uint64_t base, uint32_t len, int flags)
{
	uint64_t w = base | (uint64_t)len << 40 | (uint64_t)flags << 40;
	while (((ctx->fifo.ib_put + 1) & ctx->fifo.ib_mask) == ctx->fifo.ib_get) {
		uint32_t old = ctx->fifo.ib_get;
		ctx->fifo.ib_get = __gdev_fifo_read_reg(ctx, 0x88);
		if (old == ctx->fifo.ib_get) {
			SCHED_YIELD();
		}
	}
	ctx->fifo.ib_map[ctx->fifo.ib_put * 2] = w;
	ctx->fifo.ib_map[ctx->fifo.ib_put * 2 + 1] = w >> 32;
	ctx->fifo.ib_put++;
	ctx->fifo.ib_put &= ctx->fifo.ib_mask;
	MB(); /* is this needed? */
	ctx->dummy = ctx->fifo.ib_map[0]; /* flush writes */
	__gdev_fifo_write_reg(ctx, 0x8c, ctx->fifo.ib_put);
}

void gdev_fifo_update_get(struct gdev_ctx *ctx)
{
	uint32_t lo = __gdev_fifo_read_reg(ctx, 0x58);
	uint32_t hi = __gdev_fifo_read_reg(ctx, 0x5c);
	if (hi & 0x80000000) {
		uint64_t mg = ((uint64_t)hi << 32 | lo) & 0xffffffffffull;
		ctx->fifo.pb_get = mg - ctx->fifo.pb_base;
	} else {
		ctx->fifo.pb_get = 0;
	}
}
