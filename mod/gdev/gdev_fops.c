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

#include <linux/mm.h>
#include <linux/module.h>
#include <linux/version.h>
#include "gdev_api.h"
#include "gdev_device.h"
#include "gdev_fops.h"
#include "gdev_ioctl.h"

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
static long gdev_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
#else
static int gdev_ioctl
(struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg)
#endif
{
	Ghandle handle = filp->private_data;

	switch (cmd) {
	case GDEV_IOCTL_GET_HANDLE: /* this is not Gdev API. */
		return gdev_ioctl_get_handle(handle, arg);
	case GDEV_IOCTL_GMALLOC:
		return gdev_ioctl_gmalloc(handle, arg);
	case GDEV_IOCTL_GFREE:
		return gdev_ioctl_gfree(handle, arg);
	case GDEV_IOCTL_GMALLOC_DMA:
		return gdev_ioctl_gmalloc_dma(handle, arg);
	case GDEV_IOCTL_GFREE_DMA:
		return gdev_ioctl_gfree_dma(handle, arg);
	case GDEV_IOCTL_GMAP:
		return gdev_ioctl_gmap(handle, arg);
	case GDEV_IOCTL_GUNMAP:
		return gdev_ioctl_gunmap(handle, arg);
	case GDEV_IOCTL_GMEMCPY_TO_DEVICE:
		return gdev_ioctl_gmemcpy_to_device(handle, arg);
	case GDEV_IOCTL_GMEMCPY_TO_DEVICE_ASYNC:
		return gdev_ioctl_gmemcpy_to_device_async(handle, arg);
	case GDEV_IOCTL_GMEMCPY_FROM_DEVICE:
		return gdev_ioctl_gmemcpy_from_device(handle, arg);
	case GDEV_IOCTL_GMEMCPY_FROM_DEVICE_ASYNC:
		return gdev_ioctl_gmemcpy_from_device_async(handle, arg);
	case GDEV_IOCTL_GMEMCPY:
		return gdev_ioctl_gmemcpy(handle, arg);
	case GDEV_IOCTL_GMEMCPY_ASYNC:
		return gdev_ioctl_gmemcpy_async(handle, arg);
	case GDEV_IOCTL_GLAUNCH:
		return gdev_ioctl_glaunch(handle, arg);
	case GDEV_IOCTL_GSYNC:
		return gdev_ioctl_gsync(handle, arg);
	case GDEV_IOCTL_GBARRIER:
		return gdev_ioctl_gbarrier(handle, arg);
	case GDEV_IOCTL_GQUERY:
		return gdev_ioctl_gquery(handle, arg);
	case GDEV_IOCTL_GTUNE:
		return gdev_ioctl_gtune(handle, arg);
	case GDEV_IOCTL_GSHMGET:
		return gdev_ioctl_gshmget(handle, arg);
	case GDEV_IOCTL_GSHMAT:
		return gdev_ioctl_gshmat(handle, arg);
	case GDEV_IOCTL_GSHMDT:
		return gdev_ioctl_gshmdt(handle, arg);
	case GDEV_IOCTL_GSHMCTL:
		return gdev_ioctl_gshmctl(handle, arg);
	case GDEV_IOCTL_GREF:
		return gdev_ioctl_gref(handle, arg);
	case GDEV_IOCTL_GUNREF:
		return gdev_ioctl_gunref(handle, arg);
	case GDEV_IOCTL_GPHYSGET:
		return gdev_ioctl_gphysget(handle, arg);
	case GDEV_IOCTL_GVIRTGET:
		return gdev_ioctl_gvirtget(handle, arg);
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
			ret = remap_pfn_range(vma, start, pfn, PAGE_SIZE, PAGE_SHARED);
			if (ret < 0)
				return ret;
			start += PAGE_SIZE;
			vmalloc_area_ptr += PAGE_SIZE;
			size -= PAGE_SIZE;
		}
		
		return 0;
	}
	else {
		unsigned long pfn;
		if (virt_addr_valid(buf))
			pfn = virt_to_phys(buf) >> PAGE_SHIFT;
		else
			pfn = vmalloc_to_pfn(buf);
		return remap_pfn_range(vma, start, pfn, size, PAGE_SHARED);
	}
}

struct file_operations gdev_fops = {
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
