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

#ifndef __GDEV_DEVICE_H__
#define __GDEV_DEVICE_H__

#include "gdev_arch.h"
#include "gdev_list.h"
#include "gdev_system.h"

/**
 * maximum number of physical devices that Gdev supports
 */
#define GDEV_PHYSICAL_DEVICE_MAX_COUNT 8

/**
 * generic operation definitions
 */
#define GDEV_OP_COMPUTE 1
#define GDEV_OP_MEMCPY 2
#define GDEV_OP_MEMCPY_ASYNC 3

/**
 * Gdev device struct:
 */
struct gdev_device {
	int id; /* device ID */
	int users; /* the number of threads/processes using the device */
	int accessed; /* indicate if any process is on the device */
	int blocked; /* incidate if a process is allowed to access the device */
	uint32_t chipset;
	uint64_t mem_size;
	uint64_t mem_used;
	uint64_t dma_mem_size;
	uint64_t dma_mem_used;
	uint32_t com_bw; /* available compute bandwidth */
	uint32_t mem_bw; /* available memory bandwidth */
	uint32_t mem_sh; /* available memory space share */
	uint32_t period; /* minimum inter-arrival time (us) of replenishment. */
	uint32_t com_bw_used; /* used compute bandwidth */
	uint32_t mem_bw_used; /* used memory bandwidth */
	uint32_t com_time; /* cumulative computation time. */
	uint32_t mem_time; /* cumulative memory transfer time. */
	struct gdev_time credit_com; /* credit of compute execution */
	struct gdev_time credit_mem; /* credit of memory transfer */
	void *priv; /* private device object */
	void *compute; /* private set of compute functions */
	void *sched_com_thread; /* compute scheduler thread */
	void *sched_mem_thread; /* memory scheduler thread */
	void *credit_com_thread; /* compute credit thread */
	void *credit_mem_thread; /* memory credit thread */
	void *current_com; /* current compute execution entity */
	void *current_mem; /* current memory transfer entity */
	struct gdev_device *parent; /* only for virtual devices */
	struct gdev_list list_entry_com; /* entry to active compute list */
	struct gdev_list list_entry_mem; /* entry to active memory list */
	struct gdev_list sched_com_list; /* wait list for compute scheduling */
	struct gdev_list sched_mem_list; /* wait list for memory scheduling */
	struct gdev_list vas_list; /* list of VASes allocated to this device */
	struct gdev_list shm_list; /* list of shm users allocated to this device */
	gdev_lock_t sched_com_lock;
	gdev_lock_t sched_mem_lock;
	gdev_lock_t vas_lock;
	gdev_lock_t global_lock;
	gdev_mutex_t shm_mutex;
	gdev_mem_t *swap; /* reserved swap memory space */
};

int gdev_init_device(struct gdev_device *gdev, int id, void *priv);
void gdev_exit_device(struct gdev_device *gdev);

int gdev_init_virtual_device(struct gdev_device *gdev, int id, uint32_t weight, struct gdev_device *parent);
void gdev_exit_virtual_device(struct gdev_device*);
int gdev_getinfo_device_count(void);

extern int gdev_count;
extern int gdev_vcount;
extern struct gdev_device *gdevs;
extern struct gdev_device *gdev_vds;
extern int VCOUNT_LIST[GDEV_PHYSICAL_DEVICE_MAX_COUNT];

#endif
