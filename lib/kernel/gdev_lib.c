/*
 * Copyright (C) Shinpei Kato
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

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/unistd.h>

#include "gdev_api.h"
#include "gdev_ioctl_def.h"
#include "gdev_list.h"
#include "gdev_platform.h"

struct gdev_map_bo {
	uint64_t addr;
	uint32_t size;
	void *buf;
	struct gdev_list list_entry;
};

struct gdev_handle {
	int fd;
	struct gdev_list map_bo_list;
};

/* return "OS-space" buffer address, if @addr is associated with DMA buffer. */
static uint64_t __gdev_lookup_dma_buf(struct gdev_handle *h, uint64_t addr)
{
	struct gdev_map_bo *bo;
	uint64_t buf_addr;

	/* look up if @addr is associated with DMA buffer. */
	gdev_list_for_each (bo, &h->map_bo_list, list_entry) {
		buf_addr = (uint64_t)bo->buf;
		if ((addr >= buf_addr) && (addr < buf_addr + bo->size))
			break;
	}
	
	if (bo)
		return bo->addr + addr - buf_addr;
	else
		return 0; /* means NULL. */
}

struct gdev_handle *gopen(int minor)
{
	char devname[32];
	struct gdev_handle *h;
	int fd;

	sprintf(devname, "/dev/gdev%d", minor);
	if ((fd = open(devname, O_RDWR)) < 0)
		return NULL;

	h = (struct gdev_handle *) malloc(sizeof(*h));
	h->fd = fd;
	gdev_list_init(&h->map_bo_list, NULL);

	/* chunk size of 0x40000 seems best when using OS runtime. */
	if (gtune(h, GDEV_TUNE_MEMCPY_CHUNK_SIZE, 0x40000)) {
		return NULL;
	}

	return h;
}

int gclose(struct gdev_handle *h)
{
	int fd = h->fd;
	return close(fd);
}

uint64_t gmalloc(struct gdev_handle *h, uint64_t size)
{
	struct gdev_ioctl_mem mem;
	int fd = h->fd;

	mem.size = size;
	if(ioctl(fd, GDEV_IOCTL_GMALLOC, &mem))
		return 0;

	return mem.addr;
}

uint64_t gfree(struct gdev_handle *h, uint64_t addr)
{
	struct gdev_ioctl_mem mem;
	int fd = h->fd;

	mem.addr = addr;
	if(ioctl(fd, GDEV_IOCTL_GFREE, &mem))
		return 0;

	return mem.size;
}

void *gmalloc_dma(struct gdev_handle *h, uint64_t size)
{
	void *buf;
	struct gdev_map_bo *bo;
	struct gdev_ioctl_mem mem;
	int fd = h->fd;

	mem.addr = 0; /* will be set via ioctl() */
	mem.size = size;
	if (ioctl(fd, GDEV_IOCTL_GMALLOC_DMA, &mem))
		goto fail_gmalloc_dma;
	buf = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem.addr);
	if (buf == MAP_FAILED)
		goto fail_map;

	bo = (struct gdev_map_bo *)malloc(sizeof(*bo));
	if (!bo)
		goto fail_malloc;
	gdev_list_init(&bo->list_entry, bo);
	gdev_list_add(&bo->list_entry, &h->map_bo_list);
	bo->addr = mem.addr; /* "OS-space" buffer address */
	bo->size = size; /* could be different from mem.size */
	bo->buf = buf;

	return buf;

fail_malloc:
	munmap(buf, size);
fail_map:
	ioctl(fd, GDEV_IOCTL_GFREE_DMA, &mem);
fail_gmalloc_dma:
	return NULL;
}

uint64_t gfree_dma(struct gdev_handle *h, void *buf)
{
	struct gdev_map_bo *bo;
	struct gdev_ioctl_mem mem;
	int fd = h->fd;

	gdev_list_for_each (bo, &h->map_bo_list, list_entry) {
		if (bo->buf == buf) {
			goto free;
		}
	}
	return 0;

free:
	gdev_list_del(&bo->list_entry);
	munmap(buf, bo->size);
	mem.addr = bo->addr;

	free(bo);
	if(ioctl(fd, GDEV_IOCTL_GFREE_DMA, &mem))
		return 0;

	return mem.size;
}

void *gmap(struct gdev_handle *h, uint64_t addr, uint64_t size)
{
	struct gdev_ioctl_map map;
	struct gdev_map_bo *bo;
	void *buf;
	int fd = h->fd;

	map.addr = addr;
	map.size = size;
	map.buf = 0; /* will be set via ioctl() */
	if (ioctl(fd, GDEV_IOCTL_GMAP, &map))
		goto fail_gmap;

	buf = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map.buf);
	if (buf == MAP_FAILED)
		goto fail_map;

	bo = (struct gdev_map_bo *)malloc(sizeof(*bo));
	if (!bo)
		goto fail_malloc;
	gdev_list_init(&bo->list_entry, bo);
	gdev_list_add(&bo->list_entry, &h->map_bo_list);
	bo->addr = map.buf; /* "OS-space" buffer address */
	bo->size = size;
	bo->buf = buf;

	return buf;

fail_malloc:
	munmap(buf, size);
fail_map:
	ioctl(fd, GDEV_IOCTL_GUNMAP, &map);
fail_gmap:
	return NULL;
}

int gunmap(struct gdev_handle *h, void *buf)
{
	struct gdev_map_bo *bo;
	struct gdev_ioctl_map map;
	int fd = h->fd;

	gdev_list_for_each (bo, &h->map_bo_list, list_entry) {
		if (bo->buf == buf) {
			goto unmap;
		}
	}
	return -ENOENT;

unmap:
	gdev_list_del(&bo->list_entry);
	munmap(buf, bo->size);
	map.addr = 0; /* unused */
	map.size = 0; /* unused */
	map.buf = bo->addr; /* "OS-space" buffer address */
	free(bo);

	return ioctl(fd, GDEV_IOCTL_GUNMAP, &map);
}

static int __gmemcpy_to_device(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id, int ioctl_cmd)
{
	struct gdev_ioctl_dma dma;
	uint64_t src_addr = (uint64_t)src_buf;
	uint64_t dma_addr;
	int fd = h->fd;

	/* look up if @src_buf is allocated on DMA buffer already. */
	dma_addr = __gdev_lookup_dma_buf(h, src_addr);
	
	dma.dst_addr = dst_addr;
	if (dma_addr)
		/* this is "OS-space" buffer address associated with DMA buffer. */
		dma.src_buf = (void *)dma_addr;
	else
		/* this is "user-space" buffer address. */
		dma.src_buf = src_buf;
	dma.size = size;
	dma.id = id;

	return ioctl(fd, ioctl_cmd, &dma);
}

int gmemcpy_to_device(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size)
{
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, NULL, GDEV_IOCTL_GMEMCPY_TO_DEVICE);
}

int gmemcpy_to_device_async(struct gdev_handle *h, uint64_t dst_addr, const void *src_buf, uint64_t size, uint32_t *id)
{
	return __gmemcpy_to_device(h, dst_addr, src_buf, size, id, GDEV_IOCTL_GMEMCPY_TO_DEVICE_ASYNC);
}

static int __gmemcpy_from_device(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id, int ioctl_cmd)
{
	struct gdev_ioctl_dma dma;
	uint64_t dst_addr = (uint64_t)dst_buf;
	uint64_t dma_addr;
	int fd = h->fd;

	/* look up if @dst_buf is allocated on DMA buffer already. */
	dma_addr = __gdev_lookup_dma_buf(h, dst_addr);
	
	dma.src_addr = src_addr;
	if (dma_addr)
		/* this is "OS-space" buffer address associated with DMA buffer. */
		dma.dst_buf = (void *)dma_addr;
	else
		/* this is "user-space" buffer address. */
		dma.dst_buf = dst_buf;
	dma.size = size;
	dma.id = id;

	return ioctl(fd, ioctl_cmd, &dma);
}

int gmemcpy_from_device(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size)
{
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, NULL, GDEV_IOCTL_GMEMCPY_FROM_DEVICE);
}

int gmemcpy_from_device_async(struct gdev_handle *h, void *dst_buf, uint64_t src_addr, uint64_t size, uint32_t *id)
{
	return __gmemcpy_from_device(h, dst_buf, src_addr, size, id, GDEV_IOCTL_GMEMCPY_FROM_DEVICE_ASYNC);
}

int gmemcpy(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size)
{
	struct gdev_ioctl_dma dma;
	int fd = h->fd;

	dma.dst_addr = dst_addr;
	dma.src_addr = src_addr;
	dma.size = size;
	dma.id = NULL;

	return ioctl(fd, GDEV_IOCTL_GMEMCPY, &dma);
}

int gmemcpy_async(struct gdev_handle *h, uint64_t dst_addr, uint64_t src_addr, uint64_t size, uint32_t *id)
{
	struct gdev_ioctl_dma dma;
	int fd = h->fd;

	dma.dst_addr = dst_addr;
	dma.src_addr = src_addr;
	dma.size = size;
	dma.id = id;

	return ioctl(fd, GDEV_IOCTL_GMEMCPY_ASYNC, &dma);
}

int glaunch(struct gdev_handle *h, struct gdev_kernel *kernel, uint32_t *id)
{
	struct gdev_ioctl_launch launch;
	int fd = h->fd;

	launch.kernel = kernel;
	launch.id = id;

	return ioctl(fd, GDEV_IOCTL_GLAUNCH, &launch);
}

int gsync(struct gdev_handle *h, uint32_t id, struct gdev_time *timeout)
{
	struct gdev_ioctl_sync sync;
	int fd = h->fd;

	sync.id = id;
	sync.timeout = timeout;
	
	return ioctl(fd, GDEV_IOCTL_GSYNC, &sync);
}

int gbarrier(struct gdev_handle *h)
{
	int fd = h->fd;
	return ioctl(fd, GDEV_IOCTL_GBARRIER, NULL);
}

int gquery(struct gdev_handle *h, uint32_t type, uint64_t *result)
{
	struct gdev_ioctl_query q;
	int fd = h->fd;
	int ret;

	q.type = type;
	if ((ret = ioctl(fd, GDEV_IOCTL_GQUERY, &q)))
		return ret;
	*result = q.result;

	return 0;
}

int gtune(struct gdev_handle *h, uint32_t type, uint32_t value)
{
	struct gdev_ioctl_tune c;
	int fd = h->fd;
	int ret;

	c.type = type;
	c.value = value;
	if ((ret = ioctl(fd, GDEV_IOCTL_GTUNE, &c)))
		return ret;

	return 0;
}

int gshmget(struct gdev_handle *h, int key, uint64_t size, int flags)
{
	struct gdev_ioctl_shm s;
	int fd = h->fd;
	int ret;

	s.key = key;
	s.size = size;
	s.flags = flags;
	if ((ret = ioctl(fd, GDEV_IOCTL_GSHMGET, &s)))
		return ret;

	return 0;
}

uint64_t gshmat(struct gdev_handle *h, int id, uint64_t addr, int flags)
{
	struct gdev_ioctl_shm s;
	int fd = h->fd;
	int ret;

	s.id = id;
	s.addr = addr;
	s.flags = flags;
	if ((ret = ioctl(fd, GDEV_IOCTL_GSHMAT, &s)))
		return ret;

	return 0;
}

int gshmdt(struct gdev_handle *h, uint64_t addr)
{
	struct gdev_ioctl_shm s;
	int fd = h->fd;
	int ret;

	s.addr = addr;
	if ((ret = ioctl(fd, GDEV_IOCTL_GSHMDT, &s)))
		return ret;

	return 0;
}

int gshmctl(struct gdev_handle *h, int id, int cmd, void *buf)
{
	struct gdev_ioctl_shm s;
	int fd = h->fd;
	int ret;

	s.id = id;
	s.cmd = cmd;
	s.buf = buf;
	if ((ret = ioctl(fd, GDEV_IOCTL_GSHMCTL, &s)))
		return ret;

	return 0;
}

uint64_t gref(struct gdev_handle *hmaster, uint64_t addr, uint64_t size, struct gdev_handle *hslave)
{
	struct gdev_ioctl_ref r;
	struct gdev_ioctl_handle h;
	uint64_t dma_addr;
	int fd_master = hmaster->fd;
	int fd_slave = hslave->fd;
	int ret;

	if ((ret = ioctl(fd_slave, GDEV_IOCTL_GET_HANDLE, &h)))
		return ret;

	/* lookup if @addr is associated with DMA buffer. */
	dma_addr = __gdev_lookup_dma_buf(hmaster, addr);

	if (dma_addr)
		r.addr = dma_addr; /* "OS-space" buffer address */
	else
		r.addr = addr;
	r.size = size;
	r.handle_slave = h.handle;
	if ((ret = ioctl(fd_master, GDEV_IOCTL_GREF, &r)))
		return ret;

	return r.addr_slave;
}

int gunref(struct gdev_handle *h, uint64_t addr)
{
	struct gdev_ioctl_unref r;
	int fd = h->fd;
	int ret;

	r.addr = addr;
	if ((ret = ioctl(fd, GDEV_IOCTL_GUNREF, &r)))
		return ret;

	return 0;
}

uint64_t gphysget(struct gdev_handle *h, const void *p)
{
	struct gdev_ioctl_phys phys;
	int fd = h->fd;
	uint64_t p_addr = (uint64_t)p;
	uint64_t dma_addr;

	dma_addr = __gdev_lookup_dma_buf(h, p_addr);
	if (dma_addr)
		goto physget;

	return 0;

physget:
	phys.addr = dma_addr; /* "OS-space" buffer address */
	ioctl(fd, GDEV_IOCTL_GPHYSGET, &phys);

	return phys.phys;
}

uint64_t gvirtget(struct gdev_handle *h, const void *p)
{
	struct gdev_ioctl_virt virt;
	int fd = h->fd;
	uint64_t p_addr = (uint64_t)p;
	uint64_t dma_addr;

	dma_addr = __gdev_lookup_dma_buf(h, p_addr);
	if (dma_addr)
		goto virtget;

	return 0;

virtget:
	virt.addr = dma_addr; /* "OS-space" buffer address */
	ioctl(fd, GDEV_IOCTL_GVIRTGET, &virt);

	return virt.virt;
}

int gdevice_count(int* result)
{
	char fname[256] = "/proc/gdev/virtual_device_count";
	char buf[16];
	int minor = 0;
	FILE *fp;

	if (!(fp = fopen(fname, "r"))) {
		*result = minor;
		return -ENODEV;
	}
	if (!fgets(buf, 16, fp))
		sprintf(buf, "0");
	fclose(fp);

	sscanf(buf, "%d", &minor);

	*result = minor;
	return 0;
}
