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

#ifndef __GDEV_SYSTEM_H__
#define __GDEV_SYSTEM_H__

#ifdef __KERNEL__
#include "gdev_drv.h"
#else
#include "gdev_lib.h"
#endif
#include "gdev_platform.h"

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
int gdev_sched_wakeup(void *task);
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
void *gdev_current_com_get(struct gdev_device *gdev);
void gdev_current_com_set(struct gdev_device *gdev, void* com);
void *gdev_priv_get(struct gdev_device *gdev);
struct gdev_device *gdev_phys_get(struct gdev_device *gdev);
struct gdev_mem *gdev_swap_get(struct gdev_device *gdev);
void *gdev_compute_get(struct gdev_device *gdev);
struct gdev_sched_entity *gdev_sched_entity_alloc(int size);


#ifndef __KERNEL__ /* User-Space scheduling function.fix this  */
#ifndef GDEV_SCHED_DISABLED /* specified constant value offset */
void *gdev_attach_shms_dev(int size); 
void *gdev_attach_shms_vas(int size); 
void *gdev_attach_shms_se(int size);
void *gdev_attach_shms_mem(int size);
void gdev_next_compute(struct gdev_device *gdev);
#endif
#endif

#endif
