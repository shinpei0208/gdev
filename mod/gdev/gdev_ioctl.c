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

#include <linux/shm.h>
#include "gdev_api.h"
#include "gdev_conf.h"
#include "gdev_ioctl.h"

#define GDEV_MEMCPY_USER_DIRECT

int gdev_ioctl_get_handle(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_handle h;

	h.handle = (uint64_t)handle;

	if (copy_to_user((void __user *)arg, &h, sizeof(h)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gmalloc(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_mem m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (!(m.addr = gmalloc(handle, m.size)))
		return -ENOMEM;

	if (copy_to_user((void __user *)arg, &m, sizeof(m)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gfree(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_mem m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (!(m.size = gfree(handle, m.addr)))
		return -ENOENT;

	if (copy_to_user((void __user *)arg, &m, sizeof(m)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gmalloc_dma(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_mem m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (!(m.addr = (uint64_t)gmalloc_dma(handle, m.size)))
		return -ENOMEM;

	if (copy_to_user((void __user *)arg, &m, sizeof(m)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gfree_dma(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_mem m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (!(m.size = gfree_dma(handle, (void*)m.addr)))
		return -ENOENT;

	if (copy_to_user((void __user *)arg, &m, sizeof(m)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gmap(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_map m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (!(m.buf = (uint64_t)gmap(handle, m.addr, m.size)))
		return -ENOMEM;

	if (copy_to_user((void __user *)arg, &m, sizeof(m)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gunmap(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_map m;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;

	if (gunmap(handle, (void*)m.buf))
		return -ENOENT;

	return 0;
}

int gdev_ioctl_gmemcpy_to_device(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_dma dma;
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

int gdev_ioctl_gmemcpy_to_device_async(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_dma dma;
	int ret;
	int id;
#ifndef GDEV_MEMCPY_USER_DIRECT
	void *buf;
#endif

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

#ifdef GDEV_MEMCPY_USER_DIRECT
	ret = gmemcpy_user_to_device_async(handle, dma.dst_addr, dma.src_buf, dma.size, &id);
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

	ret = gmemcpy_to_device_async(handle, dma.dst_addr, buf, dma.size, &id);
	if (ret)
		return ret;

	if (dma.size > 0x400000)
		vfree(buf);
	else
		kfree(buf);
#endif

	if (copy_to_user((void __user *)dma.id, &id, sizeof(id)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gmemcpy_from_device(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_dma dma;
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

int gdev_ioctl_gmemcpy_from_device_async(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_dma dma;
	int ret;
	int id;
#ifndef GDEV_MEMCPY_USER_DIRECT
	void *buf;
#endif

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

#ifdef GDEV_MEMCPY_USER_DIRECT
	ret = gmemcpy_user_from_device_async(handle, dma.dst_buf, dma.src_addr, dma.size, &id);
	if (ret)
		return ret;
#else
	if (dma.size > 0x400000)
		buf = vmalloc(dma.size);
	else
		buf = kmalloc(dma.size, GFP_KERNEL);

	if (!buf)
		return -ENOMEM;

	ret = gmemcpy_from_device_async(handle, buf, dma.src_addr, dma.size, &id);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)dma.dst_buf, buf, dma.size))
		return -EFAULT;

	if (dma.size > 0x400000)
		vfree(buf);
	else
		kfree(buf);
#endif

	if (copy_to_user((void __user *)dma.id, &id, sizeof(id)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gmemcpy(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_dma dma;

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

	return gmemcpy(handle, dma.dst_addr, dma.src_addr, dma.size);
}

int gdev_ioctl_gmemcpy_async(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_dma dma;
	int id;
	int ret;

	if (copy_from_user(&dma, (void __user *)arg, sizeof(dma)))
		return -EFAULT;

	ret = gmemcpy_async(handle, dma.dst_addr, dma.src_addr, dma.size, &id);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)dma.id, &id, sizeof(id)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_glaunch(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_launch launch;
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

int gdev_ioctl_gsync(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_sync sync;
	struct gdev_time timeout;

	if (copy_from_user(&sync, (void __user *)arg, sizeof(sync)))
		return -EFAULT;

	if (!sync.timeout)
		return gsync(handle, sync.id, NULL);

	if (copy_from_user(&timeout, (void __user *)sync.timeout, sizeof(timeout)))
		return -EFAULT;

	return gsync(handle, sync.id, &timeout);
}

int gdev_ioctl_gbarrier(Ghandle handle, unsigned long arg)
{
	return gbarrier(handle);
}

int gdev_ioctl_gquery(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_query q;

	if (copy_from_user(&q, (void __user *)arg, sizeof(q)))
		return -EFAULT;

	if (gquery(handle, q.type, &q.result))
		return -EINVAL;

	if (copy_to_user((void __user *)arg, &q, sizeof(q)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gtune(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_tune c;

	if (copy_from_user(&c, (void __user *)arg, sizeof(c)))
		return -EFAULT;

	return gtune(handle, c.type, c.value);
}

int gdev_ioctl_gshmget(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_shm s;

	if (copy_from_user(&s, (void __user *)arg, sizeof(s)))
		return -EFAULT;

	return gshmget(handle, s.key, s.size, s.flags);
}

int gdev_ioctl_gshmat(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_shm s;

	if (copy_from_user(&s, (void __user *)arg, sizeof(s)))
		return -EFAULT;

	return gshmat(handle, s.id, s.addr, s.flags);
}

int gdev_ioctl_gshmdt(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_shm s;

	if (copy_from_user(&s, (void __user *)arg, sizeof(s)))
		return -EFAULT;

	return gshmdt(handle, s.addr);
}

int gdev_ioctl_gshmctl(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_shm s;
	struct shmid_ds ds;

	if (copy_from_user(&s, (void __user *)arg, sizeof(s)))
		return -EFAULT;

	if (s.buf) {
		if (copy_from_user(&ds, (void __user *)s.buf, sizeof(ds)))
			return -EFAULT;
	}
	else {
		memset(&ds, sizeof(ds), 0);
	}

	return gshmctl(handle, s.id, s.cmd, (void *)&ds);
}

int gdev_ioctl_gref(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_ref r;

	if (copy_from_user(&r, (void __user *)arg, sizeof(r)))
		return -EFAULT;

	if (!(r.addr_slave = gref(handle, r.addr, r.size, (Ghandle)r.handle_slave)))
		return -EINVAL;

	if (copy_to_user((void __user *)arg, &r, sizeof(r)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gunref(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_unref r;

	if (copy_from_user(&r, (void __user *)arg, sizeof(r)))
		return -EFAULT;

	if (gunref(handle, r.addr))
		return -EINVAL;

	return 0;
}

int gdev_ioctl_gphysget(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_phys p;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	if (!(p.phys = gphysget(handle, (void *)p.addr)))
		return -EINVAL;

	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;

	return 0;
}

int gdev_ioctl_gvirtget(Ghandle handle, unsigned long arg)
{
	struct gdev_ioctl_phys p;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	if (!(p.phys = gvirtget(handle, (void *)p.addr)))
		return -EINVAL;

	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;

	return 0;
}
