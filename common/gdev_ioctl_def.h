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

#ifndef __GDEV_IOCTL_DEF_H__
#define __GDEV_IOCTL_DEF_H__

/**
 * user-space ioctl commands:
 */
#define GDEV_IOCTL_GMALLOC 0x100
#define GDEV_IOCTL_GFREE 0x101
#define GDEV_IOCTL_GMALLOC_DMA 0x102
#define GDEV_IOCTL_GFREE_DMA 0x103
#define GDEV_IOCTL_GMEMCPY_FROM_DEVICE 0x104
#define GDEV_IOCTL_GMEMCPY_TO_DEVICE 0x105
#define GDEV_IOCTL_GMEMCPY_IN_DEVICE 0x106
#define GDEV_IOCTL_GLAUNCH 0x107
#define GDEV_IOCTL_GSYNC 0x108
#define GDEV_IOCTL_GQUERY 0x109
#define GDEV_IOCTL_GTUNE 0x110

typedef struct gdev_ioctl_mem {
	uint64_t addr;
	uint64_t size;
} gdev_ioctl_mem_t;

typedef struct gdev_ioctl_dma {
	const void *src_buf;
	void *dst_buf;
	uint64_t src_addr;
	uint64_t dst_addr;
	uint64_t size;
} gdev_ioctl_dma_t;

typedef struct gdev_ioctl_launch {
	struct gdev_kernel *kernel;
	uint32_t *id;
} gdev_ioctl_launch_t;

typedef struct gdev_ioctl_sync {
	uint32_t id;
	gdev_time_t timeout;
} gdev_ioctl_sync_t;

typedef struct gdev_ioctl_query {
	uint32_t type;
	uint32_t result;
} gdev_ioctl_query_t;

typedef struct gdev_ioctl_tune {
	uint32_t type;
	uint32_t value;
} gdev_ioctl_tune_t;

#endif
