/*
 * Copyright 2012 Shinpei Kato
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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __GDEV_SYSTEM_H__
#define __GDEV_SYSTEM_H__

#ifdef __KERNEL__
#include "gdev_drv.h"
#else
#include "gdev_lib.h"
#endif

struct gdev_device; /* prototype declaration */

/**
 * OS and user-space private types.
 */
typedef struct gdev_lock gdev_lock_t;
typedef struct gdev_mutex gdev_mutex_t;

/**
 * OS and user-space private functions.
 */
int gdev_sched_create_scheduler(struct gdev_device *gdev);
void gdev_sched_destroy_scheduler(struct gdev_device *gdev);
void *gdev_sched_get_current_task(void);
int gdev_sched_get_static_prio(void *task);
void gdev_sched_sleep(void);
void gdev_sched_wakeup(void *task);
void gdev_lock_init(gdev_lock_t *p);
void gdev_lock(gdev_lock_t *p);
void gdev_unlock(gdev_lock_t *p);
void gdev_lock_save(gdev_lock_t *p, unsigned long *flags);
void gdev_unlock_restore(gdev_lock_t *p, unsigned long *flags);
void gdev_lock_nested(gdev_lock_t *p);
void gdev_unlock_nested(gdev_lock_t *p);
void gdev_mutex_init(gdev_mutex_t *p);
void gdev_mutex_lock(gdev_mutex_t *p);
void gdev_mutex_unlock(gdev_mutex_t *p);

#ifdef __KERNEL__ /* OS functions */
#define GDEV_PRINT(fmt, arg...) printk("[gdev] " fmt, ##arg)
#ifdef GDEV_DEBUG_PRINT
#define GDEV_DPRINT(fmt, arg...)					\
	if (GDEV_DEBUG_PRINT)							\
		printk("[gdev:debug] " fmt, ##arg)
#else
#define GDEV_DPRINT(fmt, arg...)
#endif
#define MALLOC(x) vmalloc(x)
#define FREE(x) vfree(x)
#define SCHED_YIELD() yield()
#define MB() mb()
#define COPY_FROM_USER(dst, src, size) \
	copy_from_user(dst, (void __user *) src, size)
#define COPY_TO_USER(dst, src, size) \
	copy_to_user((void __user *) dst, src, size)
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
#define FREE(x) free(x)
#define SCHED_YIELD() sched_yield()
#if (__GNUC__ * 100 + __GNUC_MINOR__ >= 404)
#define MB() __sync_synchronize()
#else
#define MB()
#endif
/* should never used */
#define COPY_FROM_USER(dst, src, size) memcpy(dst, src, size) 
/* should never used */
#define COPY_TO_USER(dst, src, size) memcpy(dst, src, size)
#endif

#endif
