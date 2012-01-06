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
#include "gdev_device.h"
#include "libpscnv.h"
#include "libpscnv_ib.h"
#include "pscnv_drm.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/unistd.h>

#define GDEV_DEVICE_MAX_COUNT 32
#define PSCNV_BO_FLAGS_HOST (PSCNV_GEM_SYSRAM_SNOOP | PSCNV_GEM_MAPPABLE)

/* query a piece of the device-specific information. */
int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *result)
{
	int fd = ((unsigned long) gdev->priv & 0xffffffff); /* avoid warning :) */

	switch (type) {
	case GDEV_NVIDIA_QUERY_MP_COUNT:
		if (pscnv_getparam(fd, PSCNV_GETPARAM_MP_COUNT, result))
			return -EINVAL;
		break;
	case GDEV_QUERY_DEVICE_MEM_SIZE:
		if (pscnv_getparam(fd, PSCNV_GETPARAM_FB_SIZE, result))
			return -EINVAL;
		break;
	case GDEV_QUERY_DMA_MEM_SIZE:
		if (pscnv_getparam(fd, PSCNV_GETPARAM_AGP_SIZE, result))
			return -EINVAL;
		break;
	case GDEV_QUERY_CHIPSET:
		if (pscnv_getparam(fd, PSCNV_GETPARAM_CHIPSET_ID, result))
			return -EINVAL;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_raw_dev_open(int minor)
{
	char buf[64];
	int fd;
	struct gdev_device *gdev;

	if (!gdevs) {
		gdevs = MALLOC(sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT);
		if (!gdevs)
			return NULL;
		memset(gdevs, sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT, 0);
	}

	gdev = &gdevs[minor];

	if (gdev->users == 0) {
		sprintf(buf, DRM_DEV_NAME, DRM_DIR_NAME, minor);
		if ((fd = open(buf, O_RDWR, 0)) < 0)
			return NULL;
		gdev_init_device(gdev, minor, (void *) (unsigned long) fd);
	}		

	gdev->users++;

	return gdev;
}

/* close the specified Gdev object. */
void gdev_raw_dev_close(struct gdev_device *gdev)
{
	int fd = ((unsigned long) gdev->priv & 0xffffffff); /* avoid warning :) */
	int i;

	gdev->users--;

	if (gdev->users == 0) {
		gdev_exit_device(gdev);
		close(fd);
		for (i = 0; i < GDEV_DEVICE_MAX_COUNT; i++) {
			if (gdevs[i].users > 0)
				return;
		}
		FREE(gdevs);
		gdevs = NULL;
	}
}

/* allocate a new virual address space object. 
   pscnv_ib_chan_new() will allocate a channel object, too. */
struct gdev_vas *gdev_raw_vas_new(struct gdev_device *gdev, uint64_t size)
{
	int fd = ((unsigned long) gdev->priv & 0xffffffff); /* avoid warning :) */
	uint32_t chipset = gdev->chipset;
	struct gdev_vas *vas;
	struct pscnv_ib_chan *chan;

	if (!(vas = malloc(sizeof(*vas))))
		goto fail_vas;

    if (pscnv_ib_chan_new(fd, 0, &chan, 0, 0, 0, chipset))
        goto fail_chan;

	/* private data */
	vas->pvas = (void *) chan;
	/* VAS ID */
	vas->vid = chan->vid;

	return vas;

fail_chan:
	free(vas);
fail_vas:
	return NULL;
}

/* free the specified virtual address space object. */
void gdev_raw_vas_free(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	struct pscnv_ib_chan *chan = (struct pscnv_ib_chan *) vas->pvas;
	int fd = ((unsigned long) gdev->priv & 0xffffffff); /* avoid warning :) */

	pscnv_ib_bo_free(chan->pb);
	pscnv_ib_bo_free(chan->ib);
	munmap((void*)chan->chmap, 0x1000);
    pscnv_chan_free(chan->fd, chan->cid);
	pscnv_vspace_free(fd, chan->vid);
	free(chan);
	free(vas);
}

/* create a new GPU context object. 
   there are not many to do here, as we have already allocated a channel
   object in gdev_vas_new(), i.e., @vas holds it. */
struct gdev_ctx *gdev_raw_ctx_new
(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	struct pscnv_ib_bo *fence_bo;
	struct pscnv_ib_bo *notify_bo;
	struct pscnv_ib_chan *chan = (struct pscnv_ib_chan *) vas->pvas;
	uint32_t chipset = gdev->chipset;

	if (!(ctx = malloc(sizeof(*ctx))))
		goto fail_ctx;

	/* FIFO indirect buffer setup. */
	ctx->fifo.ib_order = chan->ib_order;
	ctx->fifo.ib_map = chan->ib->map;
	ctx->fifo.ib_bo = chan->ib;
	ctx->fifo.ib_base = chan->ib->vm_base;
	ctx->fifo.ib_mask = (1 << ctx->fifo.ib_order) - 1;
	ctx->fifo.ib_put = ctx->fifo.ib_get = 0;

	/* FIFO push buffer setup. */
	ctx->fifo.pb_order = chan->pb_order;
	ctx->fifo.pb_map = chan->pb->map;
	ctx->fifo.pb_bo = chan->pb;
	ctx->fifo.pb_base = chan->pb->vm_base;
	ctx->fifo.pb_mask = (1 << ctx->fifo.pb_order) - 1;
	ctx->fifo.pb_size = (1 << ctx->fifo.pb_order);
	ctx->fifo.pb_pos = ctx->fifo.pb_put = ctx->fifo.pb_get = 0;

	/* FIFO init: it has already been done in gdev_vas_new(). */

	/* FIFO command queue registers. */
	switch (chipset & 0xf0) {
	case 0xc0:
		ctx->fifo.regs = chan->chmap;
		break;
	default:
		goto fail_fifo_reg;
	}

	/* fence buffer. */
	if (pscnv_ib_bo_alloc(chan->fd, chan->vid, 1, PSCNV_BO_FLAGS_HOST, 0, 
						  GDEV_FENCE_BUF_SIZE, 0, &fence_bo))
		goto fail_fence_alloc;
	ctx->fence.bo = fence_bo;
	ctx->fence.map = fence_bo->map;
	ctx->fence.addr = fence_bo->vm_base;
	ctx->fence.seq = 0;

	/* interrupt buffer. */
	if (pscnv_ib_bo_alloc(chan->fd, chan->vid, 1, PSCNV_GEM_VRAM_SMALL, 0, 
						  8 /* 64 bits */, 0, &notify_bo))
		goto fail_notify_alloc;
	ctx->notify.bo = notify_bo;
	ctx->notify.addr = notify_bo->vm_base;

	/* private data */
	ctx->pctx = (void *) chan;
	/* context ID = channel ID. */
	ctx->cid = chan->cid;

	return ctx;

fail_notify_alloc:
	pscnv_ib_bo_free(fence_bo);
fail_fence_alloc:
fail_fifo_reg:
	free(ctx);
fail_ctx:
	return NULL;
}

/* destroy the specified GPU context object. */
void gdev_raw_ctx_free(struct gdev_ctx *ctx)
{
	pscnv_ib_bo_free(ctx->fence.bo);
	pscnv_ib_bo_free(ctx->notify.bo);
	free(ctx);
}

/* allocate a new memory object. */
static struct gdev_mem *__gdev_mem_alloc(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map, uint32_t flags)
{
	struct gdev_mem *mem;
	struct pscnv_ib_chan *chan = vas->pvas;
	struct pscnv_ib_bo *bo;
	uint64_t raw_size = *size;
	
	if (!(mem = (struct gdev_mem *) malloc(sizeof(*mem))))
		goto fail_mem;

	if (pscnv_ib_bo_alloc(chan->fd, chan->vid, 1, flags, 0, raw_size, 0, &bo))
		goto fail_bo;

	/* address, size, and map. */
	*addr = bo->vm_base;
	*size = bo->size;
	if (flags & PSCNV_BO_FLAGS_HOST)
		*map = bo->map;
	else
		*map = NULL;

	/* private data. */
	mem->bo = (void *) bo;

	return mem;

fail_bo:
	GDEV_PRINT("Failed to allocate PSCNV buffer object.\n");
	free(mem);
fail_mem:
	return NULL;
}

/* allocate a new device memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map)
{
	return __gdev_mem_alloc(vas, addr, size, map, PSCNV_GEM_VRAM_SMALL);
}

/* allocate a new host DMA memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc_dma(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map)
{
	return __gdev_mem_alloc(vas, addr, size, map, PSCNV_BO_FLAGS_HOST);
}

/* free the specified memory object. */
void gdev_raw_mem_free(struct gdev_mem *mem)
{
	struct pscnv_ib_bo *bo = mem->bo;

	if (pscnv_ib_bo_free(bo))
		GDEV_PRINT("Failed to free PSCNV buffer object.\n");
	free(mem);
}

/* allocate a reserved swap memory object. size may be aligned. */
struct gdev_mem *gdev_raw_swap_alloc(struct gdev_device *gdev, uint64_t size)
{
	GDEV_PRINT("Swap memory not implemented\n");
	/* To be implemented. */
	return NULL;
}

/* free the specified swap memory object. */
void gdev_raw_swap_free(struct gdev_mem *mem)
{
	GDEV_PRINT("Swap memory not implemented\n");
	/* To be implemented. */
}

/* create a new memory object sharing memory space with @mem. */
struct gdev_mem *gdev_raw_mem_share(struct gdev_vas *vas, struct gdev_mem *mem, uint64_t *addr, uint64_t *size, void **map)
{
	GDEV_PRINT("Shared memory not implemented\n");
	/* To be implemented. */
	return NULL;
}

/* destroy the memory object by just unsharing memory space. */
void gdev_raw_mem_unshare(struct gdev_mem *mem)
{
	GDEV_PRINT("Shared memory not implemented\n");
	/* To be implemented. */
}

uint32_t gdev_raw_read32(struct gdev_mem *mem, uint64_t addr)
{
	struct pscnv_ib_bo *bo = mem->bo;
	int fd = bo->fd;
	int vid = bo->vid;
	uint32_t handle = bo->handle;
	uint32_t val;

	pscnv_vm_read32(fd, vid, handle, addr, &val);

	return val;
}

void gdev_raw_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val)
{
	struct pscnv_ib_bo *bo = mem->bo;
	int fd = bo->fd;
	int vid = bo->vid;
	uint32_t handle = bo->handle;

	pscnv_vm_write32(fd, vid, handle, addr, val);
}

int gdev_raw_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size)
{
	struct pscnv_ib_bo *bo = mem->bo;
	int fd = bo->fd;
	int vid = bo->vid;
	uint32_t handle = bo->handle;

	return pscnv_vm_read(fd, vid, handle, addr, buf, size);
}

int gdev_raw_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size)
{
	struct pscnv_ib_bo *bo = mem->bo;
	int fd = bo->fd;
	int vid = bo->vid;
	uint32_t handle = bo->handle;

	return pscnv_vm_write(fd, vid, handle, addr, buf, size);
}
