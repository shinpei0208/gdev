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

#ifndef __GDEV_PLATFORM_H__
#define __GDEV_PLATFORM_H__

#ifdef __KERNEL__ /* OS functions */
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/printk.h>
#include <linux/vmalloc.h>
#include <linux/sched.h>
#else /* user-space functions */
#include <errno.h> /* ENOMEN, etc. */
#include <sched.h> /* sched_yield, etc. */
#include <stdint.h> /* uint32_t, etc.*/
#include <stdio.h> /* printf, etc. */
#include <stdlib.h> /* malloc/free, etc. */
#include <string.h> /* memcpy, etc. */
#endif

#ifdef __KERNEL__ /* OS functions */
#define GDEV_PRINT(fmt, arg...) printk("[gdev] " fmt, ##arg)
#ifdef GDEV_DEBUG_PRINT
#define GDEV_DPRINT(fmt, arg...) printk("[gdev:debug] " fmt, ##arg)
#else
#define GDEV_DPRINT(fmt, arg...)
#endif
#define MALLOC(x) vmalloc(x)
#define FREE(x) vfree(x)
#define SCHED_YIELD() schedule()
#define MB() mb()
#define COPY_FROM_USER(dst, src, size) \
		copy_from_user(dst, (void __user *) src, size)
#define COPY_TO_USER(dst, src, size) \
		copy_to_user((void __user *) dst, src, size)
#define IOREAD32(addr) ioread32((void /*__force __iomem*/ *)addr)
#define IOWRITE32(val, addr) iowrite32(val, (void /*__force __iomem*/ *)addr)

#else /* user-space functions */
#define GDEV_PRINT(fmt, arg...) fprintf(stderr, "[gdev] " fmt, ##arg)
#ifdef GDEV_DEBUG_PRINT
#define GDEV_DPRINT(fmt, arg...)					\
	if (GDEV_DEBUG_PRINT)							\
		fprintf(stderr, "[gdev:debug] " fmt, ##arg)
#else
#define GDEV_DPRINT(fmt, arg...)
#endif
#define MALLOC(x) malloc(x)
#ifdef GDEV_SCHED_DISABLED
#define FREE(x) free(x)
#else
#define FREE(x) memset(x, 0, sizeof(*x))
#endif
#ifdef SCHED_DEADLINE
#define SCHED_YIELD() if (sched_getscheduler(getpid()) != SCHED_DEADLINE) sched_yield();
#else
#define SCHED_YIELD() sched_yield()
#endif
#if (__GNUC__ * 100 + __GNUC_MINOR__ >= 404)
#define MB() __sync_synchronize()
#else
#define MB()
#endif
/* should never used */
#define COPY_FROM_USER(dst, src, size) memcpy(dst, src, size) 
/* should never used */
#define COPY_TO_USER(dst, src, size) memcpy(dst, src, size)
#define IOREAD32(addr) *(uint32_t *)(addr)
#define IOWRITE32(val, addr) *(uint32_t *)(addr) = val
#endif

#ifdef __KERNEL__ /* OS functions */
static inline char* STRDUP(const char *str) {
	size_t len;
	char *buf;
	if (!str) {
		return NULL;
	}
	len = strlen(str) + 1;
	buf = MALLOC(len);
	if (buf) {
		memcpy(buf, str, len);
	}
	return buf;
}
#else /* user-space functions */
#define STRDUP strdup
#endif

#endif  /* __GDEV_PLATFORM_H__ */
