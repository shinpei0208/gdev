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

#include "gdev_api.h"
#include "gdev_conf.h"
#include "gdev_ioctl.h"

#define GDEV_MEMCPY_USER_DIRECT

int gdev_ioctl_gmalloc(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_mem_t m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (!(m.addr = gmalloc(handle, m.size)))
		return -ENOMEM;

	if (copy_to_user((void __user *)arg, &m, sizeof(m)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gfree(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_mem_t m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	return gfree(handle, m.addr);
}

int gdev_ioctl_gmemcpy_to_device(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_dma_t dma;
	int ret;
#ifndef GDEV_MEMCPY_USER_DIRECT
	void *buf;
#endif

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

#ifdef GDEV_MEMCPY_USER_DIRECT
	ret = gmemcpy_user_to_device(handle, dma.dst_addr, dma.src_buf, dma.size);
	if (ret)
		return ret;
#else
	if (dma.size > 0x400000)
		buf = vmalloc(dma.size);
	else
		buf = kmalloc(dma.size, GFP_KERNEL);

	if (!buf)
		return -ENOMEM;

	if (copy_from_user(buf, (void __user *)dma.src_buf, dma.size))
		return -EFAULT;

	ret = gmemcpy_to_device(handle, dma.dst_addr, buf, dma.size);
	if (ret)
		return ret;

	if (dma.size > 0x400000)
		vfree(buf);
	else
		kfree(buf);
#endif

	return 0;
}

int gdev_ioctl_gmemcpy_from_device(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_dma_t dma;
	int ret;
#ifndef GDEV_MEMCPY_USER_DIRECT
	void *buf;
#endif

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

#ifdef GDEV_MEMCPY_USER_DIRECT
	ret = gmemcpy_user_from_device(handle, dma.dst_buf, dma.src_addr, dma.size);
	if (ret)
		return ret;
#else
	if (dma.size > 0x400000)
		buf = vmalloc(dma.size);
	else
		buf = kmalloc(dma.size, GFP_KERNEL);

	if (!buf)
		return -ENOMEM;

	ret = gmemcpy_from_device(handle, buf, dma.src_addr, dma.size);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)dma.dst_buf, buf, dma.size))
		return -EFAULT;

	if (dma.size > 0x400000)
		vfree(buf);
	else
		kfree(buf);
#endif

	return 0;
}

int gdev_ioctl_gmemcpy_in_device(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_dma_t dma;

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

	return gmemcpy_in_device(handle, dma.dst_addr, dma.src_addr, dma.size);
}

int gdev_ioctl_glaunch(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_launch_t launch;
	struct gdev_kernel kernel;
	uint32_t id;

	if (copy_from_user(&launch, (void __user *)arg, sizeof(launch)))
		return -EFAULT;

	if (copy_from_user(&kernel, (void __user *)launch.kernel, sizeof(kernel)))
		return -EFAULT;

	glaunch(handle, &kernel, &id);

	if (copy_to_user((void __user *)launch.id, &id, sizeof(id)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gsync(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_sync_t sync;

	if (copy_from_user(&sync, (void __user *)arg, sizeof(sync)))
		return -EFAULT;

	return gsync(handle, sync.id, &sync.timeout);
}

int gdev_ioctl_gquery(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_query_t q;

	if (copy_from_user(&q, (void __user *)arg, sizeof(q)))
		return -EFAULT;

	if (gquery(handle, q.type, &q.result))
		return -EINVAL;

	if (copy_to_user((void __user *)arg, &q, sizeof(q)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gtune(gdev_handle_t *handle, unsigned long arg)
{
	gdev_ioctl_tune_t c;

	if (copy_from_user(&c, (void __user *)arg, sizeof(c)))
		return -EFAULT;

	return gtune(handle, c.type, c.value);
}
