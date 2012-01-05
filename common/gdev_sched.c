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

#include "gdev_proto.h"
#include "gdev_sched.h"

struct gdev_sched_entity *sched_entity_ptr[GDEV_CONTEXT_MAX_COUNT];

/**
 * initialize the scheduler for the device.
 */
int gdev_init_scheduler(struct gdev_device *gdev)
{
	gdev_init_scheduler_thread(gdev);

	return 0;
}

/**
 * finalized the scheduler for the device.
 */
void gdev_exit_scheduler(struct gdev_device *gdev)
{
	gdev_exit_scheduler_thread(gdev);
}

/**
 * schedule kernel-launch calls.
 */
void gdev_schedule_launch(struct gdev_sched_entity *se)
{
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

