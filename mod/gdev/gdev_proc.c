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

#include <linux/proc_fs.h>
#include "gdev_device.h"
#include "gdev_drv.h"

#define GDEV_PROC_MAX_BUF 64

static struct proc_dir_entry *gdev_proc = NULL;
static struct proc_dir_entry *proc_dev_count;
static struct proc_dir_entry *proc_virt_dev_count;
static struct gdev_proc_vd {
	struct proc_dir_entry *dir;
	struct proc_dir_entry *com_bw;
	struct proc_dir_entry *mem_bw;
	struct proc_dir_entry *mem_sh;
	struct proc_dir_entry *period;
	struct proc_dir_entry *com_bw_used;
	struct proc_dir_entry *mem_bw_used;
	struct proc_dir_entry *phys;
} *proc_vd = NULL;
static struct semaphore proc_sem;

#if 1 //LINUX_VERSION_CODE >= KERNEL_VERSION(3,10,0)

#include <linux/module.h>
#include <linux/seq_file.h>

static int gdev_proc_val_show(struct seq_file *seq, void *offset)
{
	uint32_t *val = seq->private;

	seq_printf(seq, "%u", *val);

	return 0;
}

static ssize_t gdev_proc_val_write(struct file *file,
                                   const char __user *buffer,
                                   size_t count, loff_t *ppos)
{
	char kbuf[GDEV_PROC_MAX_BUF];
	struct seq_file *seq = file->private_data;
	uint32_t *val = seq->private;

	if (count > GDEV_PROC_MAX_BUF - 1)
		count = GDEV_PROC_MAX_BUF - 1;
	if (copy_from_user(kbuf, buffer, count)) {
		GDEV_PRINT("Failed to write /proc entry\n");
		return -EFAULT;
	}
	kbuf[count] = '\0';

	sscanf(kbuf, "%u", val); 

	return count;
}

static int gdev_proc_val_open_fs(struct inode *inode, struct file *file)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,10,0)
	return single_open(file, gdev_proc_val_show, PDE_DATA(inode));
#else
	return single_open(file, gdev_proc_val_show, PDE(inode)->data);
#endif
}

static const struct file_operations gdev_proc_dev_count_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_val_open_fs,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_virt_dev_count_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_val_open_fs,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

int gdev_proc_create(void)
{
	char name[256];

	gdev_proc = proc_mkdir("gdev", NULL);
	if (!gdev_proc) {
		GDEV_PRINT("Failed to create /proc/gdev\n");
		goto fail_proc;
	}

	/* device count */
	sprintf(name, "device_count");
	proc_dev_count = proc_create_data(name, S_IRUGO, gdev_proc,
	                                  &gdev_proc_dev_count_fops,
                                          &gdev_count);
	if (!proc_dev_count) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_proc_dev_count;
	}

	/* virtual device count */
	sprintf(name, "virtual_device_count");
	proc_virt_dev_count = proc_create_data(name, S_IRUGO, gdev_proc,
	                                       &gdev_proc_virt_dev_count_fops,
                                               &gdev_vcount);
	if (!proc_virt_dev_count) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_proc_virt_dev_count;
	}

	/* allocate virtual devices information area*/
	proc_vd = kzalloc(sizeof(*proc_vd) * gdev_vcount, GFP_KERNEL);
	if (!proc_vd) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_alloc_proc_vd;
	}
	return 0;

fail_alloc_proc_vd:
	remove_proc_entry("gdev/virtual_device_count", gdev_proc);
fail_proc_virt_dev_count:
	remove_proc_entry("gdev/device_count", gdev_proc);
fail_proc_dev_count:
	remove_proc_entry("gdev", NULL);
	gdev_proc = NULL;
fail_proc:
	return -EINVAL;
}

static int gdev_proc_util_show(struct seq_file *seq, void *offset)
{
	int count;

	down(&proc_sem);

	count = gdev_proc_val_show(seq, offset);

	up(&proc_sem);

	return count;
}

static ssize_t gdev_proc_util_write(struct file *file,
                                    const char __user *buffer,
                                    size_t count, loff_t *ppos)
{
	struct seq_file *seq = file->private_data;
	uint32_t *val = seq->private;
	uint32_t old = *val;
	int i;

	down(&proc_sem);

	count = gdev_proc_val_write(file, buffer, count, ppos);
	if (count < 0) {
		up(&proc_sem);
		return count;
	}

	if (*val > 100) {
		GDEV_PRINT("Invalid virtual device bandwidth/share %u\n", *val);
		*val = old;
	}

	/* detect any changes in memory size and reallocate swap.
	   FIXME: we don't guarantee safety when user administrators change the
	   memory utilization after virtual devices start being used. */
	for (i = 0; i < gdev_vcount; i++) {
		struct gdev_device *virt = &gdev_vds[i];
		struct gdev_device *phys = virt->parent;
		if (!phys)
			continue;
		if (virt->mem_size != phys->mem_size * virt->mem_sh / 100) {
			virt->mem_size = phys->mem_size * virt->mem_sh / 100;
			if (virt->swap) {
				uint32_t swap_size = GDEV_SWAP_MEM_SIZE * virt->mem_sh / 100;
				gdev_swap_destroy(virt);
				gdev_swap_create(virt, swap_size);
			}
		}
		if (virt->dma_mem_size != phys->dma_mem_size * virt->mem_sh / 100) {
			virt->dma_mem_size = phys->dma_mem_size * virt->mem_sh / 100;
		}
	}

	up(&proc_sem);

	return count;
}

static int gdev_proc_util_open_fs(struct inode *inode, struct file *file)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,10,0)
	return single_open(file, gdev_proc_util_show, PDE_DATA(inode));
#else
	return single_open(file, gdev_proc_util_show, PDE(inode)->data);
#endif
}

static const struct file_operations gdev_proc_com_bw_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_util_open_fs,
	.read = seq_read,
	.write = gdev_proc_util_write,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_mem_bw_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_util_open_fs,
	.read = seq_read,
	.write = gdev_proc_util_write,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_mem_sh_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_util_open_fs,
	.read = seq_read,
	.write = gdev_proc_util_write,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_period_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_val_open_fs,
	.read = seq_read,
	.write = gdev_proc_val_write,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_com_bw_used_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_util_open_fs,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_mem_bw_used_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_util_open_fs,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

static const struct file_operations gdev_proc_phys_fops = {
	.owner = THIS_MODULE,
	.open = gdev_proc_val_open_fs,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

int gdev_proc_minor_create(int vid)
{
    int i;
    char name[256];
 

    /* virtual devices information */
    sprintf(name, "vd%d", vid);
    proc_vd[vid].dir = proc_mkdir(name, gdev_proc);
    if (!proc_vd[vid].dir) {
	GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
	goto fail_proc_vd;
    }

    sprintf(name, "compute_bandwidth");
    proc_vd[vid].com_bw = proc_create_data(name, S_IFREG | S_IRUGO | S_IWUSR,
                                           proc_vd[vid].dir,
                                           &gdev_proc_com_bw_fops,
                                           &gdev_vds[vid].com_bw);
    if (!proc_vd[vid].com_bw) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sprintf(name, "memory_bandwidth");
    proc_vd[vid].mem_bw = proc_create_data(name, S_IFREG | S_IRUGO | S_IWUSR,
                                           proc_vd[vid].dir,
                                           &gdev_proc_mem_bw_fops,
                                           &gdev_vds[vid].mem_bw);
    if (!proc_vd[vid].mem_bw) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sprintf(name, "memory_share");
    proc_vd[vid].mem_sh = proc_create_data(name, S_IFREG | S_IRUGO | S_IWUSR,
                                           proc_vd[vid].dir,
                                           &gdev_proc_mem_sh_fops,
                                           &gdev_vds[vid].mem_sh);
    if (!proc_vd[vid].mem_sh) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sprintf(name, "period");
    proc_vd[vid].period = proc_create_data(name, S_IFREG | S_IRUGO | S_IWUSR,
                                           proc_vd[vid].dir,
                                           &gdev_proc_period_fops,
                                           &gdev_vds[vid].period);
    if (!proc_vd[vid].period) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sprintf(name, "compute_bandwidth_used");
    proc_vd[vid].com_bw_used = proc_create_data(name, S_IRUGO,
                                           proc_vd[vid].dir,
                                           &gdev_proc_com_bw_used_fops,
                                           &gdev_vds[vid].com_bw_used);
    if (!proc_vd[vid].com_bw_used) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sprintf(name, "memory_bandwidth_used");
    proc_vd[vid].mem_bw_used = proc_create_data(name, S_IRUGO,
                                           proc_vd[vid].dir,
                                           &gdev_proc_mem_bw_used_fops,
                                           &gdev_vds[vid].mem_bw_used);
    if (!proc_vd[vid].mem_bw_used) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sprintf(name, "phys");
    proc_vd[vid].phys = proc_create_data(name, S_IRUGO,
                                         proc_vd[vid].dir,
                                         &gdev_proc_phys_fops,
                                         &gdev_vds[vid].parent->id);
    if (!proc_vd[vid].phys) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }

    sema_init(&proc_sem, 1);
    return 0;

fail_proc_vd:
    for (i = 0; i < gdev_vcount; i++) {
	if (proc_vd[vid].dir) {
	    sprintf(name, "gdev/vd%d", i);
	    remove_proc_entry(name, gdev_proc);
	}
	if (proc_vd[vid].com_bw)
	    remove_proc_entry("compute_bandwidth", proc_vd[vid].dir);
	if (proc_vd[vid].mem_bw)
	    remove_proc_entry("memory_bandwidth", proc_vd[vid].dir);
	if (proc_vd[vid].mem_sh)
	    remove_proc_entry("memory_share", proc_vd[vid].dir);
	if (proc_vd[vid].period)
	    remove_proc_entry("period", proc_vd[vid].dir);
	if (proc_vd[i].com_bw_used)
	    remove_proc_entry("compute_bandwidth_used", proc_vd[i].dir);
	if (proc_vd[i].mem_bw_used)
	    remove_proc_entry("memory_bandwidth_used", proc_vd[i].dir);
	if (proc_vd[i].phys)
	    remove_proc_entry("phys", proc_vd[i].dir);
    }
    kfree(proc_vd);
    proc_vd = NULL;
    remove_proc_entry("gdev/virtual_device_count", gdev_proc);
    remove_proc_entry("gdev/device_count", gdev_proc);
    remove_proc_entry("gdev", NULL);
    gdev_proc = NULL;
    return -EINVAL;
}

#else

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

/* normal integer read. */
static int gdev_proc_val_read(char *page, char **start, off_t off, int count, int *eof, void *data)
{
	char kbuf[64];
	uint32_t dev_count = *((uint32_t*)data);

	sprintf(kbuf, "%u", dev_count);

	return gdev_proc_read(kbuf, page, count, eof);
}

/* normal integer write. */
static int gdev_proc_val_write(struct file *filp, const char __user *buf, unsigned long count, void *data)
{
	char kbuf[64];
	uint32_t *ptr = (uint32_t*)data;

	count = gdev_proc_write(kbuf, buf, count);
	sscanf(kbuf, "%u", ptr); 

	return count;
}

/* utilization [0,100] read. */
static int gdev_proc_util_read(char *page, char **start, off_t off, int count, int *eof, void *data)
{
	char kbuf[64];
	uint32_t util = *((uint32_t*)data);

	sprintf(kbuf, "%u", util);

	return gdev_proc_read(kbuf, page, count, eof);
}

/* utilization [0,100] write. */
static int gdev_proc_util_write(struct file *filp, const char __user *buf, unsigned long count, void *data)
{
	char kbuf[64];
	uint32_t *ptr = (uint32_t*)data;
	uint32_t old = *ptr;
	int i;

	count = gdev_proc_write(kbuf, buf, count);
	sscanf(kbuf, "%u", ptr); 
	if (*ptr > 100) {
		GDEV_PRINT("Invalid virtual device bandwidth/share %u\n", *ptr);
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
		if (virt->mem_size != phys->mem_size * virt->mem_sh / 100) {
			virt->mem_size = phys->mem_size * virt->mem_sh / 100;
			if (virt->swap) {
				uint32_t swap_size = GDEV_SWAP_MEM_SIZE * virt->mem_sh / 100;
				gdev_swap_destroy(virt);
				gdev_swap_create(virt, swap_size);
			}
		}
		if (virt->dma_mem_size != phys->dma_mem_size * virt->mem_sh / 100) {
			virt->dma_mem_size = phys->dma_mem_size * virt->mem_sh / 100;
		}
	}

	return count;
}

int gdev_proc_create(void)
{
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
	proc_dev_count->read_proc = gdev_proc_val_read;
	proc_dev_count->write_proc = NULL;
	proc_dev_count->data = (void*)&gdev_count;

	/* virtual device count */
	sprintf(name, "virtual_device_count");
	proc_virt_dev_count = create_proc_entry(name, 0644, gdev_proc);
	if (!proc_virt_dev_count) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_proc_virt_dev_count;
	}
	proc_virt_dev_count->read_proc = gdev_proc_val_read;
	proc_virt_dev_count->write_proc = NULL;
	proc_virt_dev_count->data = (void*)&gdev_vcount;



	/* allocate virtual devices information area*/
	proc_vd = kzalloc(sizeof(*proc_vd) * gdev_vcount, GFP_KERNEL);
	if (!proc_vd) {
		GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
		goto fail_alloc_proc_vd;
	}
	return 0;

fail_alloc_proc_vd:
	remove_proc_entry("gdev/virtual_device_count", gdev_proc);
fail_proc_virt_dev_count:
	remove_proc_entry("gdev/device_count", gdev_proc);
fail_proc_dev_count:
	remove_proc_entry("gdev", NULL);
	gdev_proc = NULL;
fail_proc:
	return -EINVAL;
}

int gdev_proc_minor_create(int vid)
{
    int i;
    char name[256];
 

    /* virtual devices information */
    sprintf(name, "vd%d", vid);
    proc_vd[vid].dir = proc_mkdir(name, gdev_proc);
    if (!proc_vd[vid].dir) {
	GDEV_PRINT("Failed to create /proc/gdev/%s\n", name);
	goto fail_proc_vd;
    }

    sprintf(name, "compute_bandwidth");
    proc_vd[vid].com_bw = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].com_bw) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].com_bw->read_proc = gdev_proc_util_read;
    proc_vd[vid].com_bw->write_proc = gdev_proc_util_write;
    proc_vd[vid].com_bw->data = (void*)&gdev_vds[vid].com_bw;

    sprintf(name, "memory_bandwidth");
    proc_vd[vid].mem_bw = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].mem_bw) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].mem_bw->read_proc = gdev_proc_util_read;
    proc_vd[vid].mem_bw->write_proc = gdev_proc_util_write;
    proc_vd[vid].mem_bw->data = (void*)&gdev_vds[vid].mem_bw;

    sprintf(name, "memory_share");
    proc_vd[vid].mem_sh = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].mem_sh) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].mem_sh->read_proc = gdev_proc_util_read;
    proc_vd[vid].mem_sh->write_proc = gdev_proc_util_write;
    proc_vd[vid].mem_sh->data = (void*)&gdev_vds[vid].mem_sh;

    sprintf(name, "period");
    proc_vd[vid].period = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].period) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].period->read_proc = gdev_proc_val_read;
    proc_vd[vid].period->write_proc = gdev_proc_val_write;
    proc_vd[vid].period->data = (void*)&gdev_vds[vid].period;

    sprintf(name, "compute_bandwidth_used");
    proc_vd[vid].com_bw_used = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].com_bw_used) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].com_bw_used->read_proc = gdev_proc_util_read;
    proc_vd[vid].com_bw_used->write_proc = NULL;
    proc_vd[vid].com_bw_used->data = (void*)&gdev_vds[vid].com_bw_used;

    sprintf(name, "memory_bandwidth_used");
    proc_vd[vid].mem_bw_used = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].mem_bw_used) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].mem_bw_used->read_proc = gdev_proc_util_read;
    proc_vd[vid].mem_bw_used->write_proc = NULL;
    proc_vd[vid].mem_bw_used->data = (void*)&gdev_vds[vid].mem_bw_used;

    sprintf(name, "phys");
    proc_vd[vid].phys = create_proc_entry(name, 0644, proc_vd[vid].dir);
    if (!proc_vd[vid].phys) {
	GDEV_PRINT("Failed to create /proc/gdev/vd%d/%s\n", vid, name);
	goto fail_proc_vd;
    }
    proc_vd[vid].phys->read_proc = gdev_proc_val_read;
    proc_vd[vid].phys->write_proc = NULL;
    proc_vd[vid].phys->data = (void*)&gdev_vds[vid].parent->id;

    sema_init(&proc_sem, 1);
    return 0;

fail_proc_vd:
    for (i = 0; i < gdev_vcount; i++) {
	if (proc_vd[vid].dir) {
	    sprintf(name, "gdev/vd%d", i);
	    remove_proc_entry(name, gdev_proc);
	}
	if (proc_vd[vid].com_bw)
	    remove_proc_entry("compute_bandwidth", proc_vd[vid].dir);
	if (proc_vd[vid].mem_bw)
	    remove_proc_entry("memory_bandwidth", proc_vd[vid].dir);
	if (proc_vd[vid].mem_sh)
	    remove_proc_entry("memory_share", proc_vd[vid].dir);
	if (proc_vd[vid].period)
	    remove_proc_entry("period", proc_vd[vid].dir);
	if (proc_vd[i].com_bw_used)
	    remove_proc_entry("compute_bandwidth_used", proc_vd[i].dir);
	if (proc_vd[i].mem_bw_used)
	    remove_proc_entry("memory_bandwidth_used", proc_vd[i].dir);
	if (proc_vd[i].phys)
	    remove_proc_entry("phys", proc_vd[i].dir);
    }
    kfree(proc_vd);
    proc_vd = NULL;
    remove_proc_entry("gdev/virtual_device_count", gdev_proc);
    remove_proc_entry("gdev/device_count", gdev_proc);
    remove_proc_entry("gdev", NULL);
    gdev_proc = NULL;
    return -EINVAL;
}

#endif

int gdev_proc_delete(void)
{
	int i;
	char name[256];

	if (!gdev_proc)
		goto end;

	if (!proc_vd)
		goto remove_gdev_proc_root;

	for (i = 0; i < gdev_vcount; i++) {
		if (!proc_vd[i].dir)
			continue;
		sprintf(name, "vd%d", i);
		remove_proc_entry("compute_bandwidth", proc_vd[i].dir);
		remove_proc_entry("memory_bandwidth", proc_vd[i].dir);
		remove_proc_entry("memory_share", proc_vd[i].dir);
		remove_proc_entry("period", proc_vd[i].dir);
		remove_proc_entry("compute_bandwidth_used", proc_vd[i].dir);
		remove_proc_entry("memory_bandwidth_used", proc_vd[i].dir);
		remove_proc_entry("phys", proc_vd[i].dir);
		remove_proc_entry(name, gdev_proc);
		proc_vd[i].dir = NULL;
	}

remove_gdev_proc_root:
	remove_proc_entry("virtual_device_count", gdev_proc);
	remove_proc_entry("device_count", gdev_proc);
	remove_proc_entry("gdev", NULL);
	gdev_proc = NULL;

end:
	if (proc_vd) {
		kfree(proc_vd);
		proc_vd = NULL;
	}

	return 0;
}

