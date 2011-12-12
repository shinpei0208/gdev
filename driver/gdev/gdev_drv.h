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

#ifndef __GDEV_DRV_H__
#define __GDEV_DRV_H__

#include <linux/sched.h>
#include <linux/slab.h>
#include "drmP.h"
#include "drm.h"

#define MODULE_NAME	"gdev"

#define GDEV_PRINT(fmt, arg...) printk("[gdev] " fmt, ##arg)
#define GDEV_DPRINT(fmt, arg...)				\
	if (GDEV_DEBUG_PRINT)							\
		printk("[gdev:debug] " fmt, ##arg)

/* macros for kernel-specific functions. */
#define MALLOC(x) vmalloc(x)
#define FREE(x) vfree(x)
#define SCHED_YIELD() yield()
#define MB() mb()
#define COPY_FROM_USER(dst, src, size) \
	copy_from_user(dst, (void __user *) src, size)
#define COPY_TO_USER(dst, src, size) \
	copy_to_user((void __user *) dst, src, size)
#define LOCK_INIT(ptr) gdev_lock_init_drv(ptr)
#define LOCK(ptr) gdev_lock_drv(ptr)
#define UNLOCK(ptr) gdev_unlock_drv(ptr)
#define LOCK_SAVE(ptr, flags) gdev_lock_save_drv(ptr, flags)
#define UNLOCK_RESTORE(ptr, flags) gdev_unlock_restore_drv(ptr, flags)
#define LOCK_NESTED(ptr) gdev_lock_nested_drv(ptr)
#define UNLOCK_NESTED(ptr) gdev_unlock_nested_drv(ptr)

/* typedefs for kernel-specific types. */
typedef spinlock_t gdev_lock_t;

static inline
void gdev_lock_init_drv(gdev_lock_t *lock)
{
	spin_lock_init(lock);
}

static inline
void gdev_lock_drv(gdev_lock_t *lock)
{
	spin_lock_irq(lock);
}

static inline 
void gdev_unlock_drv(gdev_lock_t *lock)
{
	spin_unlock_irq(lock);
}

static inline 
void gdev_lock_save_drv(gdev_lock_t *lock, unsigned long *flags)
{
	spin_lock_irqsave(lock, *flags);
}

static inline 
void gdev_unlock_restore_drv(gdev_lock_t *lock, unsigned long *flags)
{
	spin_unlock_irqrestore(lock, *flags);
}

static inline
void gdev_lock_nested_drv(gdev_lock_t *lock)
{
	spin_lock(lock);
}

static inline 
void gdev_unlock_nested_drv(gdev_lock_t *lock)
{
	spin_unlock(lock);
}


/**
 * Gdev init/exit functions:
 */
int gdev_major_init(struct pci_driver *);
int gdev_major_exit(void);
int gdev_minor_init(struct drm_device *);
int gdev_minor_exit(struct drm_device *);

/**
 * Gdev getinfo functions (exported to kernel modules).
 * the same information can be found in /proc/gdev/ for user-space.
 */
int gdev_getinfo_device_count(void);

extern struct gdev_device *gdevs;
extern int gdev_count;

#endif
