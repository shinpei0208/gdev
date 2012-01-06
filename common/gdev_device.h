/*
 * Copyright 2011 Shinpei Kato
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

#ifndef __GDEV_DEVICE_H__
#define __GDEV_DEVICE_H__

#include "gdev_arch.h"
#include "gdev_list.h"
#include "gdev_system.h"

/**
 * Gdev device struct:
 */
struct gdev_device {
	int id; /* device ID */
	int users; /* the number of threads/processes using the device */
	uint32_t chipset;
	uint64_t mem_size;
	uint64_t mem_used;
	uint64_t dma_mem_size;
	uint64_t dma_mem_used;
	uint32_t proc_util; /* available processor utilization */
	uint32_t mem_util; /* available memory utilization */
	void *priv; /* private device object */
	void *compute; /* private set of compute functions */
	void *sched_thread; /* scheduler thread */
	struct gdev_device *parent; /* only for virtual devices */
	struct gdev_list vas_list; /* list of VASes allocated to this device */
	gdev_lock_t vas_lock;
	gdev_mutex_t shmem_mutex;
	gdev_mem_t *swap; /* reserved swap memory space */
};

int gdev_init_device(struct gdev_device *gdev, int minor, void *priv);
void gdev_exit_device(struct gdev_device *gdev);

int gdev_init_virtual_device(struct gdev_device *gdev, int id, uint32_t proc_util, uint32_t mem_util, struct gdev_device *parent);
void gdev_exit_virtual_device(struct gdev_device*);

extern int gdev_count;
extern int gdev_vcount;
extern struct gdev_device *gdevs;
extern struct gdev_device *gdev_vds;

#endif
