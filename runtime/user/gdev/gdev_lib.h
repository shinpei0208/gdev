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

#ifndef __GDEV_LIB_H__
#define __GDEV_LIB_H__

#include <errno.h> /* ENOMEN, etc. */
#include <sched.h> /* sched_yield, etc. */
#include <stdint.h> /* uint32_t, etc.*/
#include <stdio.h> /* printf, etc. */
#include <stdlib.h> /* malloc/free, etc. */
#include <string.h> /* memcpy, etc. */

#define GDEV_DEV_GET(handle) (handle)->gdev
#define GDEV_DEV_SET(handle, dev) (handle)->gdev = (dev)
#define GDEV_VAS_GET(handle) (handle)->vas
#define GDEV_VAS_SET(handle, vas) (handle)->vas = (vas)
#define GDEV_CTX_GET(handle) (handle)->ctx
#define GDEV_CTX_SET(handle, ctx) (handle)->ctx = (ctx)
#define GDEV_DMA_GET(handle) (handle)->dma_mem
#define GDEV_DMA_SET(handle, mem) (handle)->dma_mem = (mem)
#define GDEV_PIPELINE_GET(handle) (handle)->pipeline_count
#define GDEV_PIPELINE_SET(handle, val) (handle)->pipeline_count = val
#define GDEV_CHUNK_GET(handle) (handle)->chunk_size
#define GDEV_CHUNK_SET(handle, val) (handle)->chunk_size = val
#define GDEV_MINOR_GET(handle) (handle)->dev_id 
#define GDEV_MINOR_SET(handle, val) (handle)->dev_id = val
#define GDEV_PRINT(fmt, arg...) fprintf(stderr, "[gdev] " fmt, ##arg)
#define GDEV_DPRINT(fmt, arg...)					\
	if (GDEV_DEBUG_PRINT)							\
		fprintf(stderr, "[gdev:debug] " fmt, ##arg)

#define MALLOC(x) malloc(x)
#define FREE(x) free(x)
#define SCHED_YIELD() sched_yield()
#define MB() //mb()
#define COPY_FROM_USER(dst, src, size) memcpy(dst, src, size)
#define COPY_TO_USER(dst, src, size) memcpy(dst, src, size)

#define DRM_DIR_NAME  "/dev/dri"
#define DRM_DEV_NAME  "%s/card%d"
#define DRM_IOCTL_NR(n)		_IOC_NR(n)
#define DRM_IOC_VOID		_IOC_NONE
#define DRM_IOC_READ		_IOC_READ
#define DRM_IOC_WRITE		_IOC_WRITE
#define DRM_IOC_READWRITE	_IOC_READ|_IOC_WRITE
#define DRM_IOC(dir, group, nr, size) _IOC(dir, group, nr, size)

/**
 * Gdev types:
 */
typedef struct gdev_vas gdev_vas_t;
typedef struct gdev_ctx gdev_ctx_t;
typedef struct gdev_mem gdev_mem_t;
typedef struct gdev_handle gdev_handle_t;
typedef struct gdev_device gdev_device_t;
typedef struct list_head gdev_list_t;

/**
 * Gdev handle struct:
 */
struct gdev_handle {
	gdev_device_t *gdev; /* gdev handle object. */
	gdev_vas_t *vas; /* virtual address space object. */
	gdev_ctx_t *ctx; /* device context object. */
	gdev_mem_t **dma_mem; /* host-side DMA memory object (bounce buffer). */
	uint32_t chunk_size; /* configurable memcpy chunk size. */
	int pipeline_count; /* configurable memcpy pipeline count. */
	int dev_id; /* device ID. */
};

/**
 * Gdev device struct:
 */
struct gdev_device {
	int id;
	int fd;
	int use;
	uint32_t chipset;
	void *compute; /* private set of compute functions */
};

struct list_head {
    struct list_head *next;
    struct list_head *prev;
	void *container;
};

extern int drmIoctl(int, unsigned long, void *);
extern int drmCommandWrite(int, unsigned long, void *, unsigned long);
extern int drmCommandWriteRead(int, unsigned long, void *, unsigned long);

#endif
