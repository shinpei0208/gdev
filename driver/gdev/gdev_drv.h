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

#ifndef __GDEV_DRV_H__
#define __GDEV_DRV_H__

#include <linux/sched.h>
#include <linux/slab.h>
#include "drmP.h"
#include "drm.h"

#define MODULE_NAME	"gdev"

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
#define GDEV_PRINT(fmt, arg...) printk("[gdev] " fmt, ##arg)
#define GDEV_DPRINT(fmt, arg...)				\
	if (DEBUG_PRINT)							\
		printk("[gdev:debug] " fmt, ##arg)

#define MALLOC(x) vmalloc(x)
#define FREE(x) vfree(x)
#define SCHED_YIELD() schedule_timeout(1)
#define MB() mb()
#define COPY_FROM_USER(dst, src, size) \
	copy_from_user(dst, (void __user *) src, size)
#define COPY_TO_USER(dst, src, size) \
	copy_to_user((void __user *) dst, src, size)

/**
 * Gdev types:
 */
typedef struct gdev_vas gdev_vas_t;
typedef struct gdev_ctx gdev_ctx_t;
typedef struct gdev_mem gdev_mem_t;
typedef struct gdev_handle gdev_handle_t;
typedef struct gdev_device gdev_device_t;

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
* Gdev driver module struct:
*/
struct gdev_drv {
	int count;
	dev_t dev;
	gdev_device_t *gdev;
};

/**
 * Gdev device struct:
 */
struct gdev_device {
	int id;
	int use; /* the number of threads/processes using the device. */
	struct cdev cdev; /* character device object */
	struct drm_device *drm; /* DRM device object */
	void *compute; /* private set of compute functions */
};

/**
 * Gdev init/exit functions:
 */
int gdev_major_init(struct pci_driver*);
int gdev_major_exit(void);
int gdev_minor_init(struct drm_device*);
int gdev_minor_exit(struct drm_device*);

/**
 * Export the Gdev driver module object:
 */
extern struct gdev_drv gdrv;

#endif
