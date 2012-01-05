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

#include <asm/uaccess.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/vmalloc.h>

#include "gdev_api.h"
#include "gdev_conf.h"
#include "gdev_drv.h"
#include "gdev_init.h"
#include "gdev_ioctl.h"
#include "gdev_proc.h"
#include "gdev_proto.h"

/**
 * global variables.
 */
dev_t dev;
struct cdev *cdevs; /* character devices for virtual devices */
int gdev_count; /* # of physical devices. */
int gdev_vcount; /* # of virtual devices. */
struct gdev_device *gdevs; /* physical devices */
struct gdev_device *gdev_vds; /* virtual devices */

/**
 * pointers to callback functions.
 */
void (*gdev_callback_notify)(int subc, uint32_t data);

static int __get_minor(struct file *filp)
{
	char *devname = filp->f_dentry->d_iname;
	if (strncmp(devname, "gdev", 4) == 0) {
		char *devnum = devname + 4;
		return simple_strtoul(devnum, NULL, 10);
	}
	return -EINVAL;
}

static int gdev_open(struct inode *inode, struct file *filp)
{
	int minor;
	Ghandle handle;
	
	if ((minor = __get_minor(filp)) < 0) {
		GDEV_PRINT("Could not find device.\n");
		return -EINVAL;
	}

	if (!(handle = gopen(minor))) {
		GDEV_PRINT("Out of resource.\n");
		return -ENOMEM;
	}

	filp->private_data = handle;
	
	return 0;
}

static int gdev_release(struct inode *inode, struct file *filp)
{
	Ghandle handle = filp->private_data;

	if (!handle) {
		GDEV_PRINT("Device not opened.\n");
		return -ENOENT;
	}

	return gclose(handle);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,37)
static long gdev_ioctl
(struct file *filp, unsigned int cmd, unsigned long arg)
#else
static int gdev_ioctl
(struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg)
#endif
{
	Ghandle handle = filp->private_data;

	switch (cmd) {
	case GDEV_IOCTL_GMALLOC:
		return gdev_ioctl_gmalloc(handle, arg);
	case GDEV_IOCTL_GFREE:
		return gdev_ioctl_gfree(handle, arg);
	case GDEV_IOCTL_GMALLOC_DMA:
		return gdev_ioctl_gmalloc_dma(handle, arg);
	case GDEV_IOCTL_GFREE_DMA:
		return gdev_ioctl_gfree_dma(handle, arg);
	case GDEV_IOCTL_GMEMCPY_TO_DEVICE:
		return gdev_ioctl_gmemcpy_to_device(handle, arg);
	case GDEV_IOCTL_GMEMCPY_TO_DEVICE_ASYNC:
		return gdev_ioctl_gmemcpy_to_device_async(handle, arg);
	case GDEV_IOCTL_GMEMCPY_FROM_DEVICE:
		return gdev_ioctl_gmemcpy_from_device(handle, arg);
	case GDEV_IOCTL_GMEMCPY_FROM_DEVICE_ASYNC:
		return gdev_ioctl_gmemcpy_from_device_async(handle, arg);
	case GDEV_IOCTL_GMEMCPY_IN_DEVICE:
		return gdev_ioctl_gmemcpy_in_device(handle, arg);
	case GDEV_IOCTL_GLAUNCH:
		return gdev_ioctl_glaunch(handle, arg);
	case GDEV_IOCTL_GSYNC:
		return gdev_ioctl_gsync(handle, arg);
	case GDEV_IOCTL_GQUERY:
		return gdev_ioctl_gquery(handle, arg);
	case GDEV_IOCTL_GTUNE:
		return gdev_ioctl_gtune(handle, arg);
	default:
		GDEV_PRINT("Ioctl command 0x%x is not supported.\n", cmd);
		return -EINVAL;
	}

	return 0;
}

static int gdev_mmap(struct file *filp, struct vm_area_struct *vma)
{
	void *buf;
	uint32_t size = vma->vm_end - vma->vm_start;
	unsigned long start = vma->vm_start;

	if (vma->vm_pgoff == 0) {
		/*
		 * int i = __get_minor(filp);
		 * struct gdev_device *gdev = &gdevs[i];
		 * buf = gdev->mmio_regs;
		 */
		return -EINVAL; /* mmio mapping is no longer supported. */
	}
	else {
		buf = (void*) (vma->vm_pgoff << PAGE_SHIFT);
	}

	if (size > PAGE_SIZE) {
		char *vmalloc_area_ptr = (char *)buf;
		unsigned long pfn;
		int ret;
		/* loop over all pages, map it page individually */
		while (size > 0) {
			pfn = vmalloc_to_pfn(vmalloc_area_ptr);
			if ((ret = remap_pfn_range(vma, start, pfn, PAGE_SIZE,
									   PAGE_SHARED)) < 0) {
				return ret;
			}
			start += PAGE_SIZE;
			vmalloc_area_ptr += PAGE_SIZE;
			size -= PAGE_SIZE;
		}
		
		return 0;
	}
	else {
		return remap_pfn_range(vma, start, virt_to_phys(buf) >> PAGE_SHIFT,
							   size, PAGE_SHARED);
	}
}

static struct file_operations gdev_fops = {
	.owner = THIS_MODULE,
	.open = gdev_open,
	.release = gdev_release,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,37)
	.unlocked_ioctl = gdev_ioctl,
#else
	.ioctl = gdev_ioctl,
#endif
	.mmap = gdev_mmap,
};

static void __gdev_notify_handler(int subc, uint32_t data)
{
#if 0
	struct gdev_device *gdev;
	struct gdev_sched_entity *se;
	int cid = (int)data;

	if (cid < GDEV_CONTEXT_MAX_COUNT) {
		se = sched_entity_ptr[cid];
		gdev = se->gdev;
		wake_up_process(gdev->sched_thread);
	}
#endif
}

/**
 * called for each minor physical device.
 */
int gdev_minor_init(struct drm_device *drm)
{
	int minor = drm->primary->index;

	if (minor >= gdev_count) {
		GDEV_PRINT("Could not find device %d\n", minor);
		return -EINVAL;
	}

	/* initialize the physical device. */
	gdev_init_device(&gdevs[minor], minor, drm);

	/* initialize the virtual device. 
	   when Gdev first loaded, one-to-one map physical and virtual device. */
	gdev_init_vdevice(&gdev_vds[minor], minor, 100, 100, &gdevs[minor]);

	/* initialize the scheduler for the virtual device. */
	gdev_init_scheduler(&gdev_vds[minor]);

	/* create the swap memory object, if configured, for the virtual device. */
	if (GDEV_SWAP_MEM_SIZE > 0) {
		gdev_swap_create(&gdev_vds[minor], GDEV_SWAP_MEM_SIZE);
	}

	return 0;
}

/**
 * called for each minor physical device.
 */
int gdev_minor_exit(struct drm_device *drm)
{
	int minor = drm->primary->index;
	int i;

	if (gdevs[minor].users) {
		GDEV_PRINT("Device %d has %d users\n", minor, gdevs[minor].users);
	}

	if (minor < gdev_count) {
		for (i = 0; i < gdev_vcount; i++) {
			if (gdev_vds[i].parent == &gdevs[minor]) {
				if (GDEV_SWAP_MEM_SIZE > 0) {
					gdev_swap_destroy(&gdev_vds[i]);
				}
				gdev_exit_scheduler(&gdev_vds[i]);
			}
		}
		gdev_exit_device(&gdevs[minor]);
	}
	
	return 0;
}

int gdev_major_init(struct pci_driver *pdriver)
{
	int i, ret;
	struct pci_dev *pdev = NULL;
	const struct pci_device_id *pid;

	GDEV_PRINT("Initializing module...\n");

	/* count how many physical devices are installed. */
	gdev_count = 0;
	for (i = 0; pdriver->id_table[i].vendor != 0; i++) {
		pid = &pdriver->id_table[i];
		while ((pdev =
				pci_get_subsys(pid->vendor, pid->device, pid->subvendor,
							   pid->subdevice, pdev)) != NULL) {
			if ((pdev->class & pid->class_mask) != pid->class)
				continue;
			
			gdev_count++;
		}
	}

	GDEV_PRINT("Found %d GPU physical device(s).\n", gdev_count);

	/* virtual device count. */
	gdev_vcount = GDEV_VDEVICE_COUNT;
	GDEV_PRINT("Configured %d GPU virtual device(s).\n", gdev_vcount);

	/* allocate vdev_count character devices. */
	if ((ret = alloc_chrdev_region(&dev, 0, gdev_vcount, MODULE_NAME))) {
		GDEV_PRINT("Failed to allocate module.\n");
		goto fail_alloc_chrdev;
	}

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
	for (i = 0; i < gdev_vcount; i++) {
		cdev_init(&cdevs[i], &gdev_fops);
		if ((ret = cdev_add(&cdevs[i], dev, 1))){
			GDEV_PRINT("Failed to register virtual device %d\n", i);
			goto fail_cdevs_add;
		}
	}

	/* create /proc entries. */
	if ((ret = gdev_proc_create())) {
		GDEV_PRINT("Failed to create /proc entry\n");
		goto fail_proc_create;
	}

	/* interrupt handler. */
	gdev_callback_notify = __gdev_notify_handler;

	return 0;

fail_proc_create:
fail_cdevs_add:
	for (i = 0; i < gdev_vcount; i++) {
		cdev_del(&cdevs[i]);
	}
	kfree(cdevs);
fail_alloc_cdevs:	
	kfree(gdev_vds);
fail_alloc_gdev_vds:
	kfree(gdevs);
fail_alloc_gdevs:
	unregister_chrdev_region(dev, gdev_vcount);
fail_alloc_chrdev:
	return ret;
}

int gdev_major_exit(void)
{
	int i;

	GDEV_PRINT("Exiting module...\n");

	gdev_callback_notify = NULL;

	gdev_proc_delete();

	for (i = 0; i < gdev_vcount; i++) {
		cdev_del(&cdevs[i]);
	}

	kfree(cdevs);
	kfree(gdev_vds);
	kfree(gdevs);

	unregister_chrdev_region(dev, gdev_vcount);

	return 0;
}

int gdev_getinfo_device_count(void)
{
	return gdev_vcount; /* return virtual device count. */
}
EXPORT_SYMBOL(gdev_getinfo_device_count);

/**
 * export Gdev API functions.
 */
EXPORT_SYMBOL(gopen);
EXPORT_SYMBOL(gclose);
EXPORT_SYMBOL(gmalloc);
EXPORT_SYMBOL(gfree);
EXPORT_SYMBOL(gmalloc_dma);
EXPORT_SYMBOL(gfree_dma);
EXPORT_SYMBOL(gmemcpy_from_device);
EXPORT_SYMBOL(gmemcpy_user_from_device);
EXPORT_SYMBOL(gmemcpy_to_device);
EXPORT_SYMBOL(gmemcpy_user_to_device);
EXPORT_SYMBOL(gmemcpy_in_device);
EXPORT_SYMBOL(glaunch);
EXPORT_SYMBOL(gsync);
EXPORT_SYMBOL(gquery);
EXPORT_SYMBOL(gtune);
