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

#include "gdev_sched.h"
#include "gdev_system.h"

struct gdev_sched_entity *sched_entity_ptr[GDEV_CONTEXT_MAX_COUNT];

/**
 * initialize the scheduler for the device.
 */
int gdev_init_scheduler(struct gdev_device *gdev)
{
	gdev_sched_create_scheduler(gdev);

	return 0;
}

/**
 * finalized the scheduler for the device.
 */
void gdev_exit_scheduler(struct gdev_device *gdev)
{
	gdev_sched_destroy_scheduler(gdev);
}

/**
 * create a new scheduling entity.
 */
struct gdev_sched_entity *gdev_sched_entity_create(struct gdev_device *gdev, gdev_ctx_t *ctx)
{
	struct gdev_sched_entity *se;

	if (!(se= MALLOC(sizeof(*se))))
		return NULL;

	/* set up the scheduling entity. */
	se->gdev = gdev;
	se->task = gdev_sched_get_current_task();
	se->ctx = ctx;
	se->rt_prio = GDEV_PRIO_DEFAULT;
	sched_entity_ptr[gdev_ctx_get_cid(ctx)] = se;

	return se;
}

/**
 * destroy the scheduling entity.
 */
void gdev_sched_entity_destroy(struct gdev_sched_entity *se)
{
	FREE(se);
}

/**
 * schedule kernel-launch calls.
 */
void gdev_schedule_launch(struct gdev_sched_entity *se)
{
	struct gdev_device *gdev = se->gdev;
	//struct gdev_sched_entity *se_current = gdev->
	
}

/**
 * schedule memcpy-copy calls.
 */
void gdev_schedule_memcpy(struct gdev_sched_entity *se)
{
}

/**
 * schedule next contexts.
 * invoked upon the completion of preceding contexts.
 */
void gdev_schedule_invoked(int subc, uint32_t data)
{
}

