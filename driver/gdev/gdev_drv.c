/*
 * Copyright 2011 Shinpei Kato
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

#include "gdev_api.h"
#include "gdev_conf.h"
#include "gdev_drv.h"
#include "gdev_ioctl.h"

struct gdev_drv gdrv;

static int __get_devnum(struct file *filp)
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
	int devnum;
	gdev_handle_t *handle;
	
	if ((devnum = __get_devnum(filp)) < 0) {
		GDEV_PRINT("Could not find device.\n");
		return -EINVAL;
	}

	if (!(handle = gopen(devnum))) {
		GDEV_PRINT("Out of resource.\n");
		return -ENOMEM;
	}

	filp->private_data = handle;
	
	return 0;
}

static int gdev_release(struct inode *inode, struct file *filp)
{
	gdev_handle_t *handle = filp->private_data;

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
	gdev_handle_t *handle = filp->private_data;

	switch (cmd) {
	case GDEV_IOCTL_GMALLOC:
		return gdev_ioctl_gmalloc(handle, arg);
	case GDEV_IOCTL_GFREE:
		return gdev_ioctl_gfree(handle, arg);
	case GDEV_IOCTL_GMEMCPY_FROM_DEVICE:
		return gdev_ioctl_gmemcpy_from_device(handle, arg);
	case GDEV_IOCTL_GMEMCPY_TO_DEVICE:
		return gdev_ioctl_gmemcpy_to_device(handle, arg);
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

static struct file_operations gdev_fops = {
	.owner = THIS_MODULE,
	.open = gdev_open,
	.release = gdev_release,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,37)
	.unlocked_ioctl = gdev_ioctl,
#else
	.ioctl = gdev_ioctl,
#endif
};

int gdev_minor_init(struct drm_device *drm)
{
	int ret;
	int i = drm->primary->index;
	struct gdev_device *gdev = &gdrv.gdev[i];

	if (i >= gdrv.count) {
		GDEV_PRINT("Could not find gdev%d.\n", i);
		return -EINVAL;
	}

	/* register a new character device. */
	GDEV_PRINT("Adding gdev%d.\n", i);
	cdev_init(&gdev->cdev, &gdev_fops);
	ret = cdev_add(&gdev->cdev, gdrv.dev, 1);
	if (ret < 0) {
		GDEV_PRINT("Failed to register gdev%d.\n", i);
		return ret;
	}

	/* initialize the Gdev compute engine. */
	gdev->id = i;
	gdev->drm = drm;
	gdev->use = 0;
	gdev_compute_init(gdev);

	return 0;
}

int gdev_minor_exit(struct drm_device *drm)
{
	int i = drm->primary->index;
	struct gdev_device *gdev = &gdrv.gdev[i];

	if (gdev->use) {
		GDEV_PRINT("gdev%d still has %d users.\n", i, gdev->use);
	}

	if (i < gdrv.count) {
		GDEV_PRINT("Removing gdev%d.\n", i);
		gdev->drm = NULL;
		cdev_del(&gdev->cdev);
	}
	
	return 0;
}

int gdev_major_init(struct pci_driver *pdriver)
{
	int i, ret;
	struct pci_dev *pdev = NULL;
	const struct pci_device_id *pid;

	GDEV_PRINT("Initializing module...\n");

	/* count how many devices are installed. */
	gdrv.count = 0;
	for (i = 0; pdriver->id_table[i].vendor != 0; i++) {
		pid = &pdriver->id_table[i];
		while ((pdev =
				pci_get_subsys(pid->vendor, pid->device, pid->subvendor,
							   pid->subdevice, pdev)) != NULL) {
			if ((pdev->class & pid->class_mask) != pid->class)
				continue;
			
			gdrv.count++;
		}
	}

	GDEV_PRINT("Found %d GPU device(s).\n", gdrv.count);

	ret = alloc_chrdev_region(&gdrv.dev, 0, gdrv.count, MODULE_NAME);
	if (ret < 0) {
		GDEV_PRINT("Failed to allocate module.\n");
		return ret;
	}

	/* allocate Gdev device objects. */
	gdrv.gdev = kmalloc(sizeof(struct gdev_device) * gdrv.count, GFP_KERNEL);

	return 0;
}

int gdev_major_exit(void)
{
	GDEV_PRINT("Exiting module...\n");

	kfree(gdrv.gdev);
	unregister_chrdev_region(gdrv.dev, gdrv.count);

	return 0;
}

/**
 * export Gdev API functions.
 */
EXPORT_SYMBOL(gopen);
EXPORT_SYMBOL(gclose);
EXPORT_SYMBOL(gmalloc);
EXPORT_SYMBOL(gfree);
EXPORT_SYMBOL(gmemcpy_from_device);
EXPORT_SYMBOL(gmemcpy_user_from_device);
EXPORT_SYMBOL(gmemcpy_to_device);
EXPORT_SYMBOL(gmemcpy_user_to_device);
EXPORT_SYMBOL(gmemcpy_in_device);
EXPORT_SYMBOL(glaunch);
EXPORT_SYMBOL(gsync);
EXPORT_SYMBOL(gquery);
EXPORT_SYMBOL(gtune);
