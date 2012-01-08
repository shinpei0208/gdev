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
gdev_lock_t global_sched_lock;

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
	se->prio = gdev_sched_get_static_prio(se->task);
	se->rt_prio = GDEV_PRIO_DEFAULT;
	se->launch_instances = 0;
	se->memcpy_instances = 0;
	gdev_list_init(&se->list_entry_com, (void*)se);
	gdev_list_init(&se->list_entry_mem, (void*)se);
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

#include "gdev_vsched_credit.c"

#define GDEV_VSCHED_POLICY_CREDIT
#ifdef GDEV_VSCHED_POLICY_CREDIT
struct gdev_vsched_policy *gdev_vsched = &gdev_vsched_credit;
#endif

/**
 * schedule compute calls.
 */
void gdev_schedule_compute(struct gdev_sched_entity *se)
{
	struct gdev_device *gdev = se->gdev;
	struct gdev_sched_entity *p, *tail = NULL;

resched:
	/* algorithm-specific virtual device scheduler. */
	gdev_vsched->schedule(gdev);

	/* local compute scheduler. */
	gdev_lock(&gdev->sched_com_lock);
	if (gdev->se_com_current && gdev->se_com_current != se) {
		/* insert the scheduling entity to the priority-ordered list. */
		gdev_list_for_each (p, &gdev->sched_com_list, list_entry_com) {
			if (se->prio > p->prio) {
				gdev_list_add_prev(&se->list_entry_com, &p->list_entry_com);
				break;
			}
			tail = p;
		}
		if (gdev_list_empty(&se->list_entry_com)) {
			if (tail)
				gdev_list_add_next(&se->list_entry_com, &tail->list_entry_com);
			else
				gdev_list_add(&se->list_entry_com, &gdev->sched_com_list);
		}
		gdev_unlock(&gdev->sched_com_lock);

		/* now the corresponding task will be suspended until some other tasks
		   will awaken it upon completions of their compute launches. */
		gdev_sched_sleep();

		goto resched;
	}
	else {
		/* now, let's get offloaded to the device! */
		if (se->launch_instances == 0) {
			/* record the start time. */
			gdev_time_stamp(&se->start);
		}
		se->launch_instances++;
		gdev->se_com_current = se;
		gdev_unlock(&gdev->sched_com_lock);
	}
}

/**
 * schedule the next context of compute.
 * invoked upon the completion of preceding contexts.
 */
void gdev_schedule_compute_post(struct gdev_device *gdev)
{
	struct gdev_sched_entity *se, *next;

	gdev_lock(&gdev->sched_com_lock);
	se = gdev->se_com_current;
	if (se) {
		se->launch_instances--;
		if (se->launch_instances == 0) {
			/* record the end time. */
			gdev_time_stamp(&se->end);
			gdev_time_sub(&gdev->credit_com, &se->end, &se->start);
			/* select the next context. */
			next = gdev_list_container(gdev_list_head(&gdev->sched_com_list));
			/* remove it from the waiting list. */
			if (next)
				gdev_list_del(&next->list_entry_com);
			/* now this is going to be the current context. */
			gdev->se_com_current = next;
			gdev_unlock(&gdev->sched_com_lock);
			/* wake up the next context! */
			if (next) {
				/* could be enforced when awakened. */
				gdev_sched_wakeup(next->task);
			}
		}
		else
			gdev_unlock(&gdev->sched_com_lock);
	}
	else
		gdev_unlock(&gdev->sched_com_lock);
}

/**
 * schedule memcpy-copy calls.
 */
void gdev_schedule_memory(struct gdev_sched_entity *se)
{
	//gdev_schedule_compute(se);
}

/**
 * schedule the next context of memory copy.
 * invoked upon the completion of preceding contexts.
 */
void gdev_schedule_memory_post(struct gdev_device *gdev)
{
	//gdev_schedule_compute_post(gdev);
}

void gdev_replenish_credit_compute(struct gdev_device *gdev)
{
	struct gdev_time credit, threshold;

	gdev_time_us(&credit, gdev->period * gdev->com_bw / 100);
	gdev_time_add(&gdev->credit_com, &gdev->credit_com, &credit);
	/* when the credit exceeds the threshold, all credits taken away. */
	gdev_time_us(&threshold, GDEV_CREDIT_INACTIVE_THRESHOLD);
	if (gdev_time_g(&gdev->credit_com, &threshold))
		gdev_time_us(&gdev->credit_com, 0);
}

void gdev_replenish_credit_memory(struct gdev_device *gdev)
{
	struct gdev_time credit, threshold;

	gdev_time_us(&credit, gdev->period * gdev->mem_bw / 100);
	gdev_time_add(&gdev->credit_mem, &gdev->credit_mem, &credit);
	/* when the credit exceeds the threshold, all credits taken away. */
	gdev_time_us(&threshold, GDEV_CREDIT_INACTIVE_THRESHOLD);
	if (gdev_time_g(&gdev->credit_mem, &threshold))
		gdev_time_us(&gdev->credit_mem, 0);
}
