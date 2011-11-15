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

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/unistd.h>

#include "gdev_api.h"
#include "gdev_ioctl_def.h"

gdev_handle_t *gopen(int devnum)
{
	char devname[32];
	gdev_handle_t *handle = (gdev_handle_t*) malloc(sizeof(int));

	sprintf(devname, "/dev/gdev%d", devnum);
	*handle = open(devname, O_RDWR);

	return handle;
}

int gclose(gdev_handle_t *handle)
{
	int fd = *handle;
	return close(fd);
}

uint64_t gmalloc(gdev_handle_t *handle, uint64_t size)
{
	gdev_ioctl_mem_t mem;
	int fd = *handle;

	mem.size = size;
	ioctl(fd, GDEV_IOCTL_GMALLOC, &mem);

	return mem.addr;
}

int gfree(gdev_handle_t *handle, uint64_t addr)
{
	gdev_ioctl_mem_t mem;
	int fd = *handle;

	mem.addr = addr;
	return ioctl(fd, GDEV_IOCTL_GFREE, &mem);
}

int gmemcpy_to_device
(gdev_handle_t *handle, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	gdev_ioctl_dma_t dma;
	int fd = *handle;

	dma.dst_addr = dst_addr;
	dma.src_buf = src_buf;
	dma.size = size;

	return ioctl(fd, GDEV_IOCTL_GMEMCPY_TO_DEVICE, &dma);
}

int gmemcpy_from_device
(gdev_handle_t *handle, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	gdev_ioctl_dma_t dma;
	int fd = *handle;

	dma.src_addr = src_addr;
	dma.dst_buf = dst_buf;
	dma.size = size;

	return ioctl(fd, GDEV_IOCTL_GMEMCPY_FROM_DEVICE, &dma);
}

int gmemcpy_in_device
(gdev_handle_t *handle, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	gdev_ioctl_dma_t dma;
	int fd = *handle;

	dma.dst_addr = dst_addr;
	dma.src_addr = src_addr;
	dma.size = size;

	return ioctl(fd, GDEV_IOCTL_GMEMCPY_IN_DEVICE, &dma);
}

int glaunch(gdev_handle_t *handle, struct gdev_kernel *kernel, uint32_t *id)
{
	gdev_ioctl_launch_t launch;
	int fd = *handle;

	launch.kernel = kernel;
	launch.id = id;

	return ioctl(fd, GDEV_IOCTL_GLAUNCH, &launch);
}

void gsync(gdev_handle_t *handle, uint32_t id)
{
	int fd = *handle;
	ioctl(fd, GDEV_IOCTL_GSYNC, id);
}

int gquery(gdev_handle_t *handle, uint32_t type, uint32_t *result)
{
	gdev_ioctl_query_t q;
	int fd = *handle;
	int ret;

	q.type = type;
	if ((ret = ioctl(fd, GDEV_IOCTL_GQUERY, &q)))
		return ret;
	*result = q.result;

	return 0;
}

int gtune(gdev_handle_t *handle, uint32_t type, uint32_t value)
{
	gdev_ioctl_tune_t c;
	int fd = *handle;
	int ret;

	c.type = type;
	c.value = value;
	if ((ret = ioctl(fd, GDEV_IOCTL_GTUNE, &c)))
		return ret;

	return 0;
}

