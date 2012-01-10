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

static void gdev_vsched_credit_schedule_compute(struct gdev_sched_entity *se)
{
	struct gdev_device *gdev = se->gdev;
	struct gdev_device *phys = gdev->parent;

	if (!phys)
		return;

resched:
	gdev_lock(&phys->sched_com_lock);
	if (phys->current_com && phys->current_com != gdev) {
		/* insert the scheduling entity to its local priority-ordered list. */
		gdev_lock_nested(&gdev->sched_com_lock);
		__gdev_enqueue_compute(gdev, se);
		gdev_unlock_nested(&gdev->sched_com_lock);
		gdev_unlock(&phys->sched_com_lock);

		GDEV_PRINT("Gdev#%d Sleep\n", gdev->id);

		/* now the corresponding task will be suspended until some other tasks
		   will awaken it upon completions of their compute launches. */
		gdev_sched_sleep();

		goto resched;
	}
	else {
		phys->current_com = (void *)gdev;
		gdev_unlock(&phys->sched_com_lock);

		GDEV_PRINT("Gdev#%d Run\n", gdev->id);
	}
}

static struct gdev_device *gdev_vsched_credit_select_next_compute(struct gdev_device *gdev)
{
	struct gdev_device *phys = gdev->parent;
	struct gdev_device *next;
	struct gdev_time zero;

	if (!phys)
		return gdev;

	gdev_lock(&phys->sched_com_lock);

	/* if the credit is exhausted, reinsert the device. */
	gdev_time_us(&zero, 0);
	if (gdev_time_le(&gdev->credit_com, &zero)) {
		gdev_list_del(&gdev->list_entry_com);
		gdev_list_add_tail(&gdev->list_entry_com, &phys->sched_com_list);
	}

	gdev_list_for_each(next, &phys->sched_com_list, list_entry_com) {
		gdev_lock_nested(&next->sched_com_lock);
		if (!gdev_list_empty(&next->sched_com_list)) {
			gdev_unlock_nested(&next->sched_com_lock);
			goto device_switched;
		}
		gdev_unlock_nested(&next->sched_com_lock);
	}
	next = NULL;
device_switched:
	phys->current_com = (void*)next; /* could be null */
	gdev_unlock(&phys->sched_com_lock);

	return next;
}

static void gdev_vsched_credit_replenish_compute(struct gdev_device *gdev)
{
	struct gdev_time credit, threshold;

	gdev_time_us(&credit, gdev->period * gdev->com_bw / 100);
	gdev_time_add(&gdev->credit_com, &gdev->credit_com, &credit);
	/* when the credit exceeds the threshold, all credits taken away. */
	gdev_time_us(&threshold, GDEV_CREDIT_INACTIVE_THRESHOLD);
	if (gdev_time_gt(&gdev->credit_com, &threshold))
		gdev_time_us(&gdev->credit_com, 0);
	/* when the credit exceeds the threshold in negative, even it. */
	threshold.neg = 1;
	if (gdev_time_lt(&gdev->credit_com, &threshold))
		gdev_time_us(&gdev->credit_com, 0);
}

/**
 * the Xen Credit scheduler implementation.
 */
struct gdev_vsched_policy gdev_vsched_credit = {
	.schedule_compute = gdev_vsched_credit_schedule_compute,
	.select_next_compute = gdev_vsched_credit_select_next_compute,
	.replenish_compute = gdev_vsched_credit_replenish_compute,
};
