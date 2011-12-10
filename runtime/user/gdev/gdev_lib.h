/*
* Copyright 2011 Shinpei Kato
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

#ifndef __GDEV_LIB_H__
#define __GDEV_LIB_H__

#include <errno.h> /* ENOMEN, etc. */
#include <sched.h> /* sched_yield, etc. */
#include <stdint.h> /* uint32_t, etc.*/
#include <stdio.h> /* printf, etc. */
#include <stdlib.h> /* malloc/free, etc. */
#include <string.h> /* memcpy, etc. */
#include <sys/sem.h> /* struct sembuf */

#define GDEV_PRINT(fmt, arg...) fprintf(stderr, "[gdev] " fmt, ##arg)
#define GDEV_DPRINT(fmt, arg...)					\
	if (GDEV_DEBUG_PRINT)							\
		fprintf(stderr, "[gdev:debug] " fmt, ##arg)

#define MALLOC(x) malloc(x)
#define FREE(x) free(x)
#define SCHED_YIELD() sched_yield()
#if (__GNUC__ * 100 + __GNUC_MINOR__ >= 404)
#define MB() __sync_synchronize()
#else
#define MB()
#endif
#define COPY_FROM_USER(dst, src, size) memcpy(dst, src, size)
#define COPY_TO_USER(dst, src, size) memcpy(dst, src, size)
#define LOCK(ptr) gdev_lock_user(ptr)
#define UNLOCK(ptr) gdev_unlock_user(ptr)
#define LOCK_SAVE(ptr, flags) gdev_lock_save_user(ptr, flags)
#define UNLOCK_RESTORE(ptr, flags) gdev_unlock_restore_user(ptr, flags)
#define LOCK_NESTED(ptr) gdev_lock_nested_user(ptr)
#define UNLOCK_NESTED(ptr) gdev_unlock_nested_user(ptr)

/* typedefs for user-specific types. */
typedef struct gdev_sem_struct {
	int semid;
	struct sembuf sembuf;
} gdev_lock_t;

static inline void gdev_lock_user(gdev_lock_t *lock)
{
}

static inline void gdev_unlock_user(gdev_lock_t *lock)
{
}

static inline 
void gdev_lock_save_user(gdev_lock_t *lock, unsigned long *flags)
{
}

static inline void gdev_unlock_restore_user
(gdev_lock_t *lock, unsigned long *flags)
{
}

static inline void gdev_lock_nested_user(gdev_lock_t *lock)
{
}

static inline void gdev_unlock_nested_user(gdev_lock_t *lock)
{
}

#define DRM_DIR_NAME  "/dev/dri"
#define DRM_DEV_NAME  "%s/card%d"
#define DRM_IOCTL_NR(n)		_IOC_NR(n)
#define DRM_IOC_VOID		_IOC_NONE
#define DRM_IOC_READ		_IOC_READ
#define DRM_IOC_WRITE		_IOC_WRITE
#define DRM_IOC_READWRITE	_IOC_READ|_IOC_WRITE
#define DRM_IOC(dir, group, nr, size) _IOC(dir, group, nr, size)

extern int drmIoctl(int, unsigned long, void *);
extern int drmCommandWrite(int, unsigned long, void *, unsigned long);
extern int drmCommandWriteRead(int, unsigned long, void *, unsigned long);

#endif
