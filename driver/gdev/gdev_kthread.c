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

#include <linux/kthread.h>
#include <linux/sched.h>
#include "gdev_proto.h"
#include "gdev_sched.h"

static int __gdev_sched_thread(void *__data)
{
	while (!kthread_should_stop()) {
		/* push data into the list here! */
		/*gdev_schedule_invoked();*/
		set_current_state(TASK_UNINTERRUPTIBLE);
		schedule();
	}

	return 0;
}

int gdev_init_scheduler_thread(struct gdev_device *gdev)
{
	struct sched_param sp = { .sched_priority = MAX_RT_PRIO - 1 };
	struct task_struct *p;
	char name[64];

	/* create scheduler threads. */
	sprintf(name, "gsched%d", gdev->id);
	p = kthread_create(__gdev_sched_thread, (void*)(uint64_t)gdev->id, name);
	if (p) {
		sched_setscheduler(p, SCHED_FIFO, &sp);
		wake_up_process(p);
		gdev->sched_thread = p;
	}

	return 0;
}

void gdev_exit_scheduler_thread(struct gdev_device *gdev)
{
	if (gdev->sched_thread)
		kthread_stop(gdev->sched_thread);
}
