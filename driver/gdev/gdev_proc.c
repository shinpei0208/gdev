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

#include <linux/proc_fs.h>
#include "gdev_device.h"
#include "gdev_drv.h"

#define GDEV_PROC_MAX_BUF 64

static struct proc_dir_entry *gdev_proc;
static struct proc_dir_entry *proc_dev_count;
static struct proc_dir_entry *proc_virt_dev_count;
static struct gdev_proc_vd {
	struct proc_dir_entry *dir;
	struct proc_dir_entry *processor;
	struct proc_dir_entry *memory;
} *proc_vd;
static struct semaphore proc_sem;

static int gdev_proc_read(char *kbuf, char *page, int count, int *eof)
{
	down(&proc_sem);
	sprintf(page, "%s", kbuf);
	count = strlen(page);
	*eof = 1;
	up(&proc_sem);
	return count;
}

static int gdev_proc_write(char *kbuf, const char *buf, int count)
{
	down(&proc_sem);
	if (count > GDEV_PROC_MAX_BUF - 1)
		count = GDEV_PROC_MAX_BUF - 1;
	if (copy_from_user(kbuf, buf, count)) {
		GDEV_PRINT("Failed to write /proc entry\n");
		up(&proc_sem);
		return -EFAULT;
	}
	up(&proc_sem);
	return count;
}

/* show virutal and physical device counts. */
static int device_count_read
(char *page, char **start, off_t off, int count, int *eof, void *data)
{
	char kbuf[64];
	uint32_t dev_count = *((uint32_t*)data);

	sprintf(kbuf, "%u", dev_count);

	return gdev_proc_read(kbuf, page, count, eof);
}

/* show virutal device processor/memory utilization. */
static int vd_util_read
(char *page, char **start, off_t off, int count, int *eof, void *data)
{
	char kbuf[64];
	uint32_t util = *((uint32_t*)data);

	sprintf(kbuf, "%u", util);

	return gdev_proc_read(kbuf, page, count, eof);
}

/* set virutal device processor/memory utilization. */
static int vd_util_write
(struct file *filp, const char __user *buf, unsigned long count, void *data)
{
	char kbuf[64];
	uint32_t *ptr = (uint32_t*)data;
	uint32_t old = *ptr;
	int i;

	count = gdev_proc_write(kbuf, buf, count);
	sscanf(kbuf, "%u", ptr); 
	if (*ptr > 100) {
		GDEV_PRINT("Invalid virtual device utilization %u\n", *ptr);
		*ptr = old;
	}

	/* detect any changes in memory size and reallocate swap.
	   FIXME: we don't guarantee safety when user administrators change the
	   memory utilization after virtual devices start being used. */
	for (i = 0; i < gdev_vcount; i++) {
		struct gdev_device *virt = &gdev_vds[i];
		struct gdev_device *phys = virt->parent;
		if (!phys)
			continue;
		if (virt->mem_size != phys->mem_size * virt->mem_util / 100) {
			virt->mem_size = phys->mem_size * virt->mem_util / 100;
			if (virt->swap) {
				uint32_t swap_size = GDEV_SWAP_MEM_SIZE * virt->mem_util / 100;
				gdev_swap_destroy(virt);
				gdev_swap_create(virt, swap_size);
			}
		}
		if (virt->dma_mem_size != phys->dma_mem_size * virt->mem_util / 100) {
			virt->dma_mem_size = phys->dma_mem_size * virt->mem_util / 100;
		}
	}

	return count;
}

int gdev_proc_create(void)
{
	int i;
	char name[256];

	gdev_proc = proc_mkdir("gdev", NULL);
	if (!gdev_proc) {
		GDEV_PRINT("Failed to create /proc/gdev\n");
		goto fail_proc;
	}

	/* device count */
	sprintf(name, "device_count");
	proc_dev_count = create_proc_entry(name, 0644, gdev_proc);
	if (!proc_dev_count) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_proc_dev_count;
	}
	proc_dev_count->read_proc = device_count_read;
	proc_dev_count->write_proc = NULL;
	proc_dev_count->data = (void*)&gdev_count;

	/* virtual device count */
	sprintf(name, "virtual_device_count");
	proc_virt_dev_count = create_proc_entry(name, 0644, gdev_proc);
	if (!proc_virt_dev_count) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_proc_virt_dev_count;
	}
	proc_virt_dev_count->read_proc = device_count_read;
	proc_virt_dev_count->write_proc = NULL;
	proc_virt_dev_count->data = (void*)&gdev_vcount;

	/* virtual devices information */
	proc_vd = kzalloc(sizeof(*proc_vd) * gdev_vcount, GFP_KERNEL);
	if (!proc_vd) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_alloc_proc_vd;
	}
	for (i = 0; i < gdev_vcount; i++) {
		sprintf(name, "vd%d", i);
		proc_vd[i].dir = proc_mkdir(name, gdev_proc);
		if (!proc_vd[i].dir) {
			GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
			goto fail_proc_vd;
		}

		sprintf(name, "processor");
		proc_vd[i].processor = create_proc_entry(name, 0644, proc_vd[i].dir);
		if (!proc_vd[i].processor) {
			GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", i, name);
			goto fail_proc_vd;
		}
		proc_vd[i].processor->read_proc = vd_util_read;
		proc_vd[i].processor->write_proc = vd_util_write;
		proc_vd[i].processor->data = (void*)&gdev_vds[i].proc_util;

		sprintf(name, "memory");
		proc_vd[i].memory = create_proc_entry(name, 0644, proc_vd[i].dir);
		if (!proc_vd[i].memory) {
			GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", i, name);
			goto fail_proc_vd;
		}
		proc_vd[i].memory->read_proc = vd_util_read;
		proc_vd[i].memory->write_proc = vd_util_write;
		proc_vd[i].memory->data = (void*)&gdev_vds[i].mem_util;
	}

	sema_init(&proc_sem, 1);

	return 0;

fail_proc_vd:
	for (i = 0; i < gdev_vcount; i++) {
		if (proc_vd[i].dir) {
			sprintf(name, "gdev/vd%d", i);
			remove_proc_entry(name, gdev_proc);
		}
		if (proc_vd[i].processor)
			remove_proc_entry("processor", proc_vd[i].dir);
		if (proc_vd[i].memory)
			remove_proc_entry("memory", proc_vd[i].memory);
	}
	kfree(proc_vd);
fail_alloc_proc_vd:
	remove_proc_entry("gdev/virtual_device_count", gdev_proc);
fail_proc_virt_dev_count:
	remove_proc_entry("gdev/device_count", gdev_proc);
fail_proc_dev_count:
	remove_proc_entry("gdev", NULL);
fail_proc:
	return -EINVAL;
}

int gdev_proc_delete(void)
{
	int i;
	char name[256];

	for (i = 0; i < gdev_vcount; i++) {
		sprintf(name, "gdev/vd%d", i);
		remove_proc_entry(name, gdev_proc);
		remove_proc_entry("processor", proc_vd[i].dir);
		remove_proc_entry("memory", proc_vd[i].memory);
	}
	kfree(proc_vd);

	remove_proc_entry("gdev/virtual_device_count", gdev_proc);
	remove_proc_entry("gdev/device_count", gdev_proc);
	remove_proc_entry("gdev", NULL);

	return 0;
}

