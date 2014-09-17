/*
 * Copyright (C) Yusuke Suzuki
 *
 * Keio University
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

#ifndef __GDEV_BENCH_H__
#define __GDEV_BENCH_H__
#include "gdev_time.h"
#include "gdev_platform.h"

struct gdev_bench {
	struct gdev_time elapsed;
	int opened;
};

static inline void gdev_bench_open(struct gdev_bench* bench)
{
	bench->opened = 1;
	gdev_time_stamp(&bench->elapsed);
}

static inline void gdev_bench_close(struct gdev_bench* bench, const char* prefix)
{
	struct gdev_time finish = { 0 };
	struct gdev_time elapsed = { 0 };

	gdev_time_stamp(&finish);
	gdev_time_sub(&elapsed, &finish, &bench->elapsed);

	bench->elapsed = elapsed;
	bench->opened = 0;
	if (prefix) {
		const long long unsigned ms = gdev_time_to_ms(&bench->elapsed);
		const long long unsigned us = gdev_time_to_us(&bench->elapsed);
		GDEV_PRINT("[%s] %llums (%lluus)\n", prefix, ms, us);
	}
}

#define GDEV_BENCH(bench) \
    for (gdev_bench_open(bench); (bench)->opened; gdev_bench_close(bench, __func__))

#endif
