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

#include <asm/uaccess.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "gdev_api.h"
#include "gdev_conf.h"
#include "gdev_device.h"
#include "gdev_drv.h"
#include "gdev_fops.h"
#include "gdev_interface.h"
#include "gdev_proc.h"
#include "gdev_sched.h"

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("Gdev");
MODULE_AUTHOR("Shinpei Kato");

#define MODULE_NAME	"gdev"

/**
 * global variables.
 */
static dev_t dev;
static struct class *dev_class;
static int cdevs_registered = 0;
static struct cdev *cdevs; /* character devices for virtual devices */

/**
 * interrupt notify handler
 */
static void __gdev_notify_handler(int op, uint32_t data)
{
	struct gdev_device *gdev;
	struct gdev_sched_entity *se;
	int cid = (int)data;

	if (cid < GDEV_CONTEXT_MAX_COUNT) {
		se = sched_entity_ptr[cid];
		gdev = se->gdev;
		switch (op) {
		case GDEV_OP_COMPUTE:
		case GDEV_OP_MEMCPY: /* memcpy is synchronous with compute */
			wake_up_process(gdev->sched_com_thread);
			break;
		case GDEV_OP_MEMCPY_ASYNC:
			wake_up_process(gdev->sched_mem_thread);
			break;
		default:
			GDEV_PRINT("Unknown operation %d\n", op);
		}
	}
	else
		GDEV_PRINT("Unknown context %d\n", cid);
}

static int __gdev_sched_com_thread(void *__data)
{
	struct gdev_device *gdev = (struct gdev_device*)__data;

	GDEV_PRINT("Gdev#%d compute scheduler running\n", gdev->id);
	gdev->sched_com_thread = current;

	while (!kthread_should_stop()) {
		set_current_state(TASK_UNINTERRUPTIBLE);
		schedule();
		if (gdev->users)
			gdev_select_next_compute(gdev);
	}

	return 0;
}

static int __gdev_sched_mem_thread(void *__data)
{
	struct gdev_device *gdev = (struct gdev_device*)__data;

	GDEV_PRINT("Gdev#%d memory scheduler running\n", gdev->id);
	gdev->sched_mem_thread = current;

	while (!kthread_should_stop()) {
		set_current_state(TASK_UNINTERRUPTIBLE);
		schedule();
		if (gdev->users)
			gdev_select_next_memory(gdev);
	}

	return 0;
}

static void __gdev_credit_handler(unsigned long __data)
{
	struct task_struct *p = (struct task_struct *)__data;
	wake_up_process(p);
}

static int __gdev_credit_com_thread(void *__data)
{
	struct gdev_device *gdev = (struct gdev_device*)__data;
	struct gdev_time now, last, elapse, interval;
	struct timer_list timer;
	unsigned long effective_jiffies;

	GDEV_PRINT("Gdev#%d compute reserve running\n", gdev->id);

	setup_timer_on_stack(&timer, __gdev_credit_handler, (unsigned long)current);

	gdev_time_us(&interval, GDEV_UPDATE_INTERVAL);
	gdev_time_stamp(&last);
	effective_jiffies = jiffies;

	while (!kthread_should_stop()) {
		gdev_replenish_credit_compute(gdev);
		mod_timer(&timer, effective_jiffies + usecs_to_jiffies(gdev->period));
		set_current_state(TASK_UNINTERRUPTIBLE);
		schedule();
		effective_jiffies = jiffies;

		gdev_lock(&gdev->sched_com_lock);
		gdev_time_stamp(&now);
		gdev_time_sub(&elapse, &now, &last);
		gdev->com_bw_used = gdev->com_time * 100 / gdev_time_to_us(&elapse);
		if (gdev->com_bw_used > 100)
			gdev->com_bw_used = 100;
		if (gdev_time_ge(&elapse, &interval)) {
			gdev->com_time = 0;
			gdev_time_stamp(&last);
		}
		gdev_unlock(&gdev->sched_com_lock);
	}

	local_irq_enable();
	if (timer_pending(&timer)) {
		del_timer_sync(&timer);
	}
	local_irq_disable();
	destroy_timer_on_stack(&timer);

	return 0;
}

static int __gdev_credit_mem_thread(void *__data)
{
	struct gdev_device *gdev = (struct gdev_device*)__data;
	struct gdev_time now, last, elapse, interval;
	struct timer_list timer;
	unsigned long effective_jiffies;

	GDEV_PRINT("Gdev#%d memory reserve running\n", gdev->id);

	setup_timer_on_stack(&timer, __gdev_credit_handler, (unsigned long)current);

	gdev_time_us(&interval, GDEV_UPDATE_INTERVAL);
	gdev_time_stamp(&last);
	effective_jiffies = jiffies;

	while (!kthread_should_stop()) {
		gdev_replenish_credit_memory(gdev);
		mod_timer(&timer, effective_jiffies + usecs_to_jiffies(gdev->period));
		set_current_state(TASK_UNINTERRUPTIBLE);
		schedule();
		effective_jiffies = jiffies;

		gdev_lock(&gdev->sched_mem_lock);
		gdev_time_stamp(&now);
		gdev_time_sub(&elapse, &now, &last);
		gdev->mem_bw_used = gdev->mem_time * 100 / gdev_time_to_us(&elapse);
		if (gdev->mem_bw_used > 100)
			gdev->mem_bw_used = 100;
		if (gdev_time_ge(&elapse, &interval)) {
			gdev->mem_time = 0;
			gdev_time_stamp(&last);
		}
		gdev_unlock(&gdev->sched_mem_lock);
	}

	local_irq_enable();
	if (timer_pending(&timer)) {
		del_timer_sync(&timer);
	}
	local_irq_disable();
	destroy_timer_on_stack(&timer);

	return 0;
}

int gdev_sched_create_scheduler(struct gdev_device *gdev)
{
	struct sched_param sp = { .sched_priority = MAX_RT_PRIO - 1 };
	struct task_struct *sched_com, *sched_mem;
	struct task_struct *credit_com, *credit_mem;
	char name[64];

	/* create compute and memory scheduler threads. */
	sprintf(name, "gschedc%d", gdev->id);
	sched_com = kthread_create(__gdev_sched_com_thread, (void*)gdev, name);
	if (sched_com) {
		sched_setscheduler(sched_com, SCHED_FIFO, &sp);
		wake_up_process(sched_com);
		gdev->sched_com_thread = sched_com;
	}
	sprintf(name, "gschedm%d", gdev->id);
	sched_mem = kthread_create(__gdev_sched_mem_thread, (void*)gdev, name);
	if (sched_mem) {
		sched_setscheduler(sched_mem, SCHED_FIFO, &sp);
		wake_up_process(sched_mem);
		gdev->sched_mem_thread = sched_mem;
	}

	/* create compute and memory credit replenishment threads. */
	sprintf(name, "gcreditc%d", gdev->id);
	credit_com = kthread_create(__gdev_credit_com_thread, (void*)gdev, name);
	if (credit_com) {
		sched_setscheduler(credit_com, SCHED_FIFO, &sp);
		wake_up_process(credit_com);
		gdev->credit_com_thread = credit_com;
	}
	sprintf(name, "gcreditm%d", gdev->id);
	credit_mem = kthread_create(__gdev_credit_mem_thread, (void*)gdev, name);
	if (credit_mem) {
		sched_setscheduler(credit_mem, SCHED_FIFO, &sp);
		wake_up_process(credit_mem);
		gdev->credit_mem_thread = credit_mem;
	}

	return 0;
}

void gdev_sched_destroy_scheduler(struct gdev_device *gdev)
{
	if (gdev->credit_mem_thread) {
		kthread_stop(gdev->credit_mem_thread);
		gdev->credit_mem_thread = NULL;
	}
	if (gdev->credit_com_thread) {
		kthread_stop(gdev->credit_com_thread);
		gdev->credit_com_thread = NULL;
	}
	if (gdev->sched_mem_thread) {
		kthread_stop(gdev->sched_mem_thread);
		gdev->sched_mem_thread = NULL;	
}
	if (gdev->sched_com_thread) {
		kthread_stop(gdev->sched_com_thread);
		gdev->sched_com_thread = NULL;
	}
	schedule_timeout_uninterruptible(usecs_to_jiffies(gdev->period));
}

void *gdev_sched_get_current_task(void)
{
	return (void*)current;
}

int gdev_sched_get_static_prio(void *task)
{
	struct task_struct *p = (struct task_struct *)task;
	return p->static_prio;
}

void gdev_sched_sleep(void)
{
	set_current_state(TASK_UNINTERRUPTIBLE);
	schedule();
}

int gdev_sched_wakeup(void *task)
{
	if (!wake_up_process(task)) {
		schedule_timeout_interruptible(1);
		if (!wake_up_process(task))
			return -EINVAL;
	}
	return 0;
}

void gdev_lock_init(struct gdev_lock *p)
{
	spin_lock_init(&p->lock);
}

void gdev_lock(struct gdev_lock *p)
{
	spin_lock_irq(&p->lock);
}

void gdev_unlock(struct gdev_lock *p)
{
	spin_unlock_irq(&p->lock);
}

void gdev_lock_save(struct gdev_lock *p, unsigned long *pflags)
{
	spin_lock_irqsave(&p->lock, *pflags);
}

void gdev_unlock_restore(struct gdev_lock *p, unsigned long *pflags)
{
	spin_unlock_irqrestore(&p->lock, *pflags);
}

void gdev_lock_nested(struct gdev_lock *p)
{
	spin_lock(&p->lock);
}

void gdev_unlock_nested(struct gdev_lock *p)
{
	spin_unlock(&p->lock);
}

void gdev_mutex_init(struct gdev_mutex *p)
{
	mutex_init(&p->mutex);
}

void gdev_mutex_lock(struct gdev_mutex *p)
{
	mutex_lock(&p->mutex);
}

void gdev_mutex_unlock(struct gdev_mutex *p)
{
	mutex_unlock(&p->mutex);
}

void* gdev_current_com_get(struct gdev_device *gdev)
{
   	return !gdev->current_com? NULL:(void*)gdev->current_com;
}
void gdev_current_com_set(struct gdev_device *gdev,void *com)
{
    	gdev->current_com = com;
}

void* gdev_compute_get(struct gdev_device *gdev)
{
 	return gdev->compute;
}

void* gdev_priv_get(struct gdev_device *gdev)
{
        return gdev->priv;
}

struct gdev_device *gdev_phys_get(struct gdev_device *gdev)
{
    	return gdev->parent;
}

struct gdev_mem *gdev_swap_get(struct gdev_device *gdev)
{
    	return gdev->swap;
}

struct gdev_sched_entity *gdev_sched_entity_alloc(int size){
	return (struct gdev_sched_entity *)MALLOC( sizeof(struct gdev_sched_entity));
}

/**
 * called for each minor physical device.
 */
int gdev_minor_init(int physid)
{
	int i, j;
	struct drm_device *drm;

	if (gdev_drv_getdrm(physid, &drm)) {
		GDEV_PRINT("Could not find device %d\n", physid);
		return -EINVAL;
	}

	/* initialize the physical device. */
	gdev_init_device(&gdevs[physid], physid, drm);

	j = 0;
	for (i = 0; i < physid; i++)
		j += VCOUNT_LIST[i];

	for (i = j; i < j + VCOUNT_LIST[physid]; i++) {
		/* initialize the virtual device. when Gdev first loaded, one-to-one
		   map physical and virtual device. */
		if (i == j) /* the first virtual device in a physical device */
			gdev_init_virtual_device(&gdev_vds[i], i, 100, &gdevs[physid]);
		else
			gdev_init_virtual_device(&gdev_vds[i], i, 0, &gdevs[physid]);

		/* initialize the local scheduler for each virtual device. */
#ifndef GDEV_SCHED_DISABLED
		gdev_init_scheduler(&gdev_vds[i]);
#endif
		/* create /proc/gdev/vd%d entries  */
		gdev_proc_minor_create(i);

		device_create(dev_class, NULL, MKDEV(MAJOR(dev), i), NULL,
			      MODULE_NAME"%d", i);
	}

	return 0;
}

/**
 * called for each minor physical device.
 */
int gdev_minor_exit(int physid)
{
	int i, j;

	if (!gdevs) {
		GDEV_PRINT("Failed to exit minor device %d, "
		           "major already exited\n", physid);
		return -EINVAL;
	}

	if (gdevs[physid].users) {
		GDEV_PRINT("Device %d has %d users\n", physid, gdevs[physid].users);
	}

	if (physid < gdev_count) {

		for (i = 0, j = 0; i < physid; i++)
			j += VCOUNT_LIST[i];

		for (i = 0; i < gdev_vcount; i++, j++) {

			device_destroy(dev_class, MKDEV(MAJOR(dev), j));

			if (gdev_vds[i].parent == &gdevs[physid]) {
#ifndef GDEV_SCHED_DISABLED
				gdev_exit_scheduler(&gdev_vds[i]);
#endif
				gdev_exit_virtual_device(&gdev_vds[i]);
			}
		}
		gdev_exit_device(&gdevs[physid]);
	}
	
	return 0;
}

int gdev_major_init(void)
{
	int i, major, ret;

	/* get the number of physical devices. */
	if (gdev_drv_getdevice(&gdev_count)) {
		GDEV_PRINT("Failed to get device count\n");
		ret = -EINVAL;
		goto fail_getdevice;
	}

	GDEV_PRINT("Found %d physical device(s).\n", gdev_count);

	/* get the number of virtual devices. */
	gdev_vcount = 0;
	for (i = 0; i < gdev_count; i++)
		gdev_vcount += VCOUNT_LIST[i];

	GDEV_PRINT("Configured %d virtual device(s).\n", gdev_vcount);

	/* allocate vdev_count character devices. */
	if ((ret = alloc_chrdev_region(&dev, 0, gdev_vcount, MODULE_NAME))) {
		GDEV_PRINT("Failed to allocate module.\n");
		goto fail_alloc_chrdev;
	}
	cdevs_registered = 1;

	/* allocate Gdev physical device objects. */
	if (!(gdevs = kzalloc(sizeof(*gdevs) * gdev_count, GFP_KERNEL))) {
		ret = -ENOMEM;
		goto fail_alloc_gdevs;
	}
	/* allocate Gdev virtual device objects. */
	if (!(gdev_vds = kzalloc(sizeof(*gdev_vds) * gdev_vcount, GFP_KERNEL))) {
		ret = -ENOMEM;
		goto fail_alloc_gdev_vds;
	}
	/* allocate character device objects. */
	if (!(cdevs = kzalloc(sizeof(*cdevs) * gdev_vcount, GFP_KERNEL))) {
		ret = -ENOMEM;
		goto fail_alloc_cdevs;
	}

	/* register character devices. */
	major = MAJOR(dev);
	for (i = 0; i < gdev_vcount; i++) {
		cdev_init(&cdevs[i], &gdev_fops);
		cdevs[i].owner = THIS_MODULE;
		if ((ret = cdev_add(&cdevs[i], MKDEV(major, i), 1))){
			GDEV_PRINT("Failed to register virtual device %d\n", i);
			goto fail_cdevs_add;
		}
	}

	/* create /proc entries. */
	if ((ret = gdev_proc_create())) {
		GDEV_PRINT("Failed to create /proc entry\n");
		goto fail_proc_create;
	}

	/* set interrupt handler. */
#ifndef GDEV_SCHED_DISABLED
	gdev_drv_setnotify(__gdev_notify_handler);
#endif

	dev_class = class_create(THIS_MODULE, MODULE_NAME);

	return 0;

fail_proc_create:
fail_cdevs_add:
	for (i = 0; i < gdev_vcount; i++) {
		cdev_del(&cdevs[i]);
	}
	kfree(cdevs);
	cdevs = NULL;
fail_alloc_cdevs:	
	kfree(gdev_vds);
	gdev_vds = NULL;
fail_alloc_gdev_vds:
	kfree(gdevs);
	gdevs = NULL;
fail_alloc_gdevs:
	unregister_chrdev_region(dev, gdev_vcount);
	cdevs_registered = 0;
fail_alloc_chrdev:
fail_getdevice:
	return ret;
}

int gdev_major_exit(void)
{
	int i;

	 class_destroy(dev_class);

#ifndef GDEV_SCHED_DISABLED
	gdev_drv_unsetnotify(__gdev_notify_handler);
#endif

	gdev_proc_delete();

	if (!cdevs)
		goto end;
	for (i = 0; i < gdev_vcount; i++) {
		cdev_del(&cdevs[i]);
	}
	kfree(cdevs);
	cdevs = NULL;

	if (!gdev_vds)
		goto end;
	kfree(gdev_vds);
	gdev_vds = NULL;

	if (!gdevs)
		goto end;
	kfree(gdevs);
	gdevs = NULL;

	if (!cdevs_registered)
		goto end;
	unregister_chrdev_region(dev, gdev_vcount);
	cdevs_registered = 0;

end:
	return 0;
}

int gdev_getinfo_device_count(void)
{
	return gdev_vcount; /* return virtual device count. */
}

/**
 * export Gdev API functions.
 */
EXPORT_SYMBOL(gopen);
EXPORT_SYMBOL(gclose);
EXPORT_SYMBOL(gmalloc);
EXPORT_SYMBOL(gfree);
EXPORT_SYMBOL(gmalloc_dma);
EXPORT_SYMBOL(gfree_dma);
EXPORT_SYMBOL(gmap);
EXPORT_SYMBOL(gunmap);
EXPORT_SYMBOL(gmemcpy_to_device);
EXPORT_SYMBOL(gmemcpy_to_device_async);
EXPORT_SYMBOL(gmemcpy_user_to_device);
EXPORT_SYMBOL(gmemcpy_user_to_device_async);
EXPORT_SYMBOL(gmemcpy_from_device);
EXPORT_SYMBOL(gmemcpy_from_device_async);
EXPORT_SYMBOL(gmemcpy_user_from_device);
EXPORT_SYMBOL(gmemcpy_user_from_device_async);
EXPORT_SYMBOL(gmemcpy);
EXPORT_SYMBOL(gmemcpy_async);
EXPORT_SYMBOL(glaunch);
EXPORT_SYMBOL(gsync);
EXPORT_SYMBOL(gbarrier);
EXPORT_SYMBOL(gquery);
EXPORT_SYMBOL(gtune);
EXPORT_SYMBOL(gshmget);
EXPORT_SYMBOL(gshmat);
EXPORT_SYMBOL(gshmdt);
EXPORT_SYMBOL(gshmctl);
EXPORT_SYMBOL(gref);
EXPORT_SYMBOL(gunref);
EXPORT_SYMBOL(gphysget);
EXPORT_SYMBOL(gvirtget);
EXPORT_SYMBOL(gdevice_count);

static int __init gdev_module_init(void)
{
	int i;

	GDEV_PRINT("Loading module...\n");

	if (gdev_major_init()) {
		GDEV_PRINT("Failed to initialize major device(s)\n");
		goto end;
	}

	for (i = 0; i < gdev_count; i++) {
		if (gdev_minor_init(i)) {
			for (i = i - 1; i >= 0; i--)
				gdev_minor_exit(i);
			gdev_major_exit();
			goto end;
		}
	}

end:
	return 0;
}

static void __exit gdev_module_exit(void)
{
	int i;

	GDEV_PRINT("Unloading module...\n");

	for (i = 0; i < gdev_count; i++) {
		gdev_minor_exit(i);
	}

	gdev_major_exit();
}

module_init(gdev_module_init);
module_exit(gdev_module_exit);
