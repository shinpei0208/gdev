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

#ifndef __GDEV_IOCTL_DEF_H__
#define __GDEV_IOCTL_DEF_H__

/**
 * utility ioctl commands:
 */
#define GDEV_IOCTL_GET_HANDLE 0x10

/**
 * user-space ioctl commands for Gdev API:
 */
#define GDEV_IOCTL_GMALLOC 0x100
#define GDEV_IOCTL_GFREE 0x101
#define GDEV_IOCTL_GMALLOC_DMA 0x102
#define GDEV_IOCTL_GFREE_DMA 0x103
#define GDEV_IOCTL_GMAP 0x104
#define GDEV_IOCTL_GUNMAP 0x105
#define GDEV_IOCTL_GMEMCPY_TO_DEVICE 0x106
#define GDEV_IOCTL_GMEMCPY_TO_DEVICE_ASYNC 0x107
#define GDEV_IOCTL_GMEMCPY_FROM_DEVICE 0x108
#define GDEV_IOCTL_GMEMCPY_FROM_DEVICE_ASYNC 0x109
#define GDEV_IOCTL_GMEMCPY 0x110
#define GDEV_IOCTL_GMEMCPY_ASYNC 0x111
#define GDEV_IOCTL_GLAUNCH 0x112
#define GDEV_IOCTL_GSYNC 0x113
#define GDEV_IOCTL_GBARRIER 0x114
#define GDEV_IOCTL_GQUERY 0x115
#define GDEV_IOCTL_GTUNE 0x116
#define GDEV_IOCTL_GSHMGET 0x117
#define GDEV_IOCTL_GSHMAT 0x118
#define GDEV_IOCTL_GSHMDT 0x119
#define GDEV_IOCTL_GSHMCTL 0x120
#define GDEV_IOCTL_GREF 0x121
#define GDEV_IOCTL_GUNREF 0x122
#define GDEV_IOCTL_GPHYSGET 0x123
#define GDEV_IOCTL_GVIRTGET 0x124

struct gdev_ioctl_handle {
	uint64_t handle;
};

struct gdev_ioctl_mem {
	uint64_t addr;
	uint64_t size;
};

struct gdev_ioctl_dma {
	const void *src_buf;
	void *dst_buf;
	uint64_t src_addr;
	uint64_t dst_addr;
	uint64_t size;
	uint32_t *id;
};

struct gdev_ioctl_launch {
	struct gdev_kernel *kernel;
	uint32_t *id;
};

struct gdev_ioctl_sync {
	uint32_t id;
	struct gdev_time *timeout;
};

struct gdev_ioctl_query {
	uint32_t type;
	uint64_t result;
};

struct gdev_ioctl_tune {
	uint32_t type;
	uint32_t value;
};

struct gdev_ioctl_shm {
	int key;
	int id;
	int flags;
	int cmd;
	uint64_t addr;
	uint64_t size;
	void *buf;
};

struct gdev_ioctl_map {
	uint64_t addr;
	uint64_t buf;
	uint64_t size;
};

struct gdev_ioctl_ref {
	uint64_t addr;
	uint64_t size;
	uint64_t handle_slave;
	uint64_t addr_slave;
};

struct gdev_ioctl_unref {
	uint64_t addr;
};

struct gdev_ioctl_phys {
	uint64_t addr;
	uint64_t phys;
};

struct gdev_ioctl_virt {
	uint64_t addr;
	uint64_t virt;
};

#endif
