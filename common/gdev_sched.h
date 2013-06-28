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

#ifndef __GDEV_SCHED_H__
#define __GDEV_SCHED_H__

#include "gdev_device.h"
#include "gdev_time.h"

/**
 * Queueing methods:
 * SDQ: Single Device Queue
 * MRQ: Multiple Resource Queues
 */
#define GDEV_SCHED_SDQ /*GDEV_SCHED_MRQ */

/**
 * priority levels.
 */
#define GDEV_PRIO_MAX 40
#define GDEV_PRIO_MIN 0
#define GDEV_PRIO_DEFAULT 20

/**
 * virtual device period/threshold.
 */
#define GDEV_PERIOD_DEFAULT 30000 /* microseconds */
#define GDEV_CREDIT_INACTIVE_THRESHOLD GDEV_PERIOD_DEFAULT
#define GDEV_UPDATE_INTERVAL 1000000

/**
 * scheduling properties.
 */
#define GDEV_INSTANCES_LIMIT 32

struct gdev_sched_entity {
	struct gdev_device *gdev; /* associated Gdev (virtual) device */
	void *task; /* private task structure */
	gdev_ctx_t *ctx; /* holder context */
	int prio; /* general priority */
	int rt_prio; /* real-time priority */
	struct gdev_list list_entry_com; /* entry to compute scheduler list */
	struct gdev_list list_entry_mem; /* entry to memory scheduler list */
	struct gdev_time last_tick_com; /* last tick of compute execution */
	struct gdev_time last_tick_mem; /* last tick of memory transfer */
	int launch_instances;
	int memcpy_instances;
};

struct gdev_vsched_policy {
	void (*schedule_compute)(struct gdev_sched_entity *se);
	struct gdev_device *(*select_next_compute)(struct gdev_device *gdev);
	void (*replenish_compute)(struct gdev_device *gdev);
	void (*schedule_memory)(struct gdev_sched_entity *se);
	struct gdev_device *(*select_next_memory)(struct gdev_device *gdev);
	void (*replenish_memory)(struct gdev_device *gdev);
};

int gdev_init_scheduler(struct gdev_device *gdev);
void gdev_exit_scheduler(struct gdev_device *gdev);

struct gdev_sched_entity *gdev_sched_entity_create(struct gdev_device *gdev, gdev_ctx_t *ctx);
void gdev_sched_entity_destroy(struct gdev_sched_entity *se);

void gdev_schedule_compute(struct gdev_sched_entity *se);
void gdev_select_next_compute(struct gdev_device *gdev);
void gdev_schedule_memory(struct gdev_sched_entity *se);
void gdev_select_next_memory(struct gdev_device *gdev);
void gdev_replenish_credit_compute(struct gdev_device *gdev);
void gdev_replenish_credit_memory(struct gdev_device *gdev);

extern struct gdev_sched_entity *sched_entity_ptr[GDEV_CONTEXT_MAX_COUNT];
extern gdev_lock_t global_sched_lock;

#endif
