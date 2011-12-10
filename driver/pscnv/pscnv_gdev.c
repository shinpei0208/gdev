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

#include "gdev_drv.h"
#include "gdev_list.h"
#include "gdev_nvidia.h"
#include "gdev_proto.h"
#include "nouveau_drv.h"
#include "pscnv_chan.h"
#include "pscnv_fifo.h"
#include "pscnv_gem.h"
#include "pscnv_ioctl.h"
#include "pscnv_mem.h"
#include "pscnv_vm.h"

extern uint32_t *nvc0_fifo_ctrl_ptr(struct drm_device *, struct pscnv_chan *);

/* query device-specific information. */
int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *result)
{
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_pscnv_getparam getparam;

	switch (type) {
	case GDEV_NVIDIA_QUERY_MP_COUNT:
		getparam.param = PSCNV_GETPARAM_MP_COUNT;
		pscnv_ioctl_getparam(drm, &getparam, NULL);
		*result = getparam.value;
		break;
	case GDEV_NVIDIA_QUERY_DEVICE_MEM_SIZE:
		getparam.param = PSCNV_GETPARAM_FB_SIZE;
		pscnv_ioctl_getparam(drm, &getparam, NULL);
		*result = getparam.value;
		break;
	case GDEV_NVIDIA_QUERY_DMA_MEM_SIZE:
		getparam.param = PSCNV_GETPARAM_AGP_SIZE;
		pscnv_ioctl_getparam(drm, &getparam, NULL);
		*result = getparam.value;
		break;
	case GDEV_NVIDIA_QUERY_CHIPSET:
		getparam.param = PSCNV_GETPARAM_CHIPSET_ID;
		pscnv_ioctl_getparam(drm, &getparam, NULL);
		*result = getparam.value;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_raw_dev_open(int minor)
{
	struct gdev_device *gdev = &gdevs[minor];

	gdev->users++;

	return gdev;
}

/* close the specified Gdev object. */
void gdev_raw_dev_close(struct gdev_device *gdev)
{
	gdev->users--;
}

/* allocate a new virual address space object. */
struct gdev_vas *gdev_raw_vas_new(struct gdev_device *gdev, uint64_t size)
{
	struct gdev_vas *vas;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct pscnv_vspace *vspace;

	if (!(vas = kzalloc(sizeof(*vas), GFP_KERNEL)))
		goto fail_vas;

	if (!(vspace = pscnv_vspace_new(drm, size, 0, 0)))
		goto fail_vspace;

	vspace->filp = NULL; /* we don't need vspace->filp in Gdev. */
	vas->pvas = (void *) vspace; /* driver private object. */

	return vas;

fail_vspace:
	kfree(vas);
fail_vas:
	return NULL;
}

/* free the specified virtual address space object. */
void gdev_raw_vas_free(struct gdev_vas *vas)
{
	struct pscnv_vspace *vspace = vas->pvas;

	vspace->filp = NULL;
	pscnv_vspace_unref(vspace);

	kfree(vas);
}

/* create a new GPU context object. */
struct gdev_ctx *gdev_raw_ctx_new
(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *priv = drm->dev_private;
	uint32_t chipset = priv->chipset;
	struct pscnv_vspace *vspace = vas->pvas; 
	struct pscnv_chan *chan;
	struct pscnv_bo *ib_bo, *pb_bo, *fence_bo;
	struct pscnv_mm_node *ib_mm, *pb_mm, *fence_mm;
	int i, ret;

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		goto fail_ctx;

	chan = pscnv_chan_new(drm, vspace, 0);
	if (!chan)
		goto fail_chan;

	/* we don't need chan->filp in Gdev. */
	chan->filp = NULL;

    if ((chipset & 0xf0) != 0xc0) {
		/* TODO: set up vdma here! */
	}

	/* FIFO indirect buffer setup. */
	ctx->fifo.ib_order = 9; /* it's hardcoded. */
	ib_bo = pscnv_mem_alloc(drm, 8 << ctx->fifo.ib_order, 
							PSCNV_GEM_SYSRAM_SNOOP, 0, 0);
	if (!ib_bo) {
		goto fail_ib;
	}
	ret = pscnv_vspace_map(vspace, ib_bo, GDEV_VAS_USER_START, 
						   GDEV_VAS_USER_END, 0, &ib_mm);
	if (ret)
		goto fail_ibmap;
	ctx->fifo.ib_map = vmap(ib_bo->pages, ib_bo->size >> PAGE_SHIFT, 0, 
							PAGE_KERNEL);
	ctx->fifo.ib_bo = ib_bo;
	ctx->fifo.ib_base = ib_mm->start;
	ctx->fifo.ib_mask = (1 << ctx->fifo.ib_order) - 1;
	ctx->fifo.ib_put = ctx->fifo.ib_get = 0;

	/* FIFO push buffer setup. */
	ctx->fifo.pb_order = 20; /* it's hardcoded. */
	pb_bo = pscnv_mem_alloc(drm, 1 << ctx->fifo.pb_order, 
							PSCNV_GEM_SYSRAM_SNOOP, 0, 0);
	if (!pb_bo)
		goto fail_pb;
	ret = pscnv_vspace_map(vspace, pb_bo, GDEV_VAS_USER_START, 
						   GDEV_VAS_USER_END, 0, &pb_mm);
	if (ret)
		goto fail_pbmap;
	ctx->fifo.pb_map = vmap(pb_bo->pages, pb_bo->size >> PAGE_SHIFT, 0, 
							PAGE_KERNEL);
	ctx->fifo.pb_bo = pb_bo;
	ctx->fifo.pb_base = pb_mm->start;
	ctx->fifo.pb_mask = (1 << ctx->fifo.pb_order) - 1;
	ctx->fifo.pb_size = (1 << ctx->fifo.pb_order);
	ctx->fifo.pb_pos = ctx->fifo.pb_put = ctx->fifo.pb_get = 0;

	/* FIFO init. */
	ret = priv->fifo->chan_init_ib(chan, 0, 0, 1, 
								   ctx->fifo.ib_base, ctx->fifo.ib_order);
	if (ret)
		goto fail_fifo_init;

	/* FIFO command queue registers. */
	switch (chipset & 0xf0) {
	case 0xc0:
		ctx->fifo.regs = nvc0_fifo_ctrl_ptr(drm, chan);
		break;
	default:
		goto fail_fifo_reg;
	}

	/* fences init. */
	fence_bo = pscnv_mem_alloc(drm, PAGE_SIZE, PSCNV_GEM_SYSRAM_SNOOP, 0, 0);
	if (!fence_bo)
		goto fail_fence_alloc;
	ret = pscnv_vspace_map(vspace, fence_bo, GDEV_VAS_USER_START, 
						   GDEV_VAS_USER_END, 0, &fence_mm);
	if (ret)
		goto fail_fence_map;
	ctx->fence.bo = fence_bo;
	ctx->fence.map = kmap(fence_bo->pages[0]); /* assume < PAGE_SIZE */
	ctx->fence.addr = fence_mm->start;
	for (i = 0; i < GDEV_FENCE_COUNT; i++) {
		ctx->fence.sequence[i] = 0;
	}

	/* private data. */
	ctx->pctx = (void *) chan;

	return ctx;
	
fail_fence_map:
	pscnv_mem_free(fence_bo);
fail_fence_alloc:
fail_fifo_reg:
fail_fifo_init:
	vunmap(ctx->fifo.pb_map);
	pscnv_vspace_unmap(vspace, pb_mm->start);
fail_pbmap:
	pscnv_mem_free(pb_bo);
fail_pb:
	vunmap(ctx->fifo.ib_map);
	pscnv_vspace_unmap(vspace, ib_mm->start);
fail_ibmap:
	pscnv_mem_free(ib_bo);
fail_ib:
	chan->filp = NULL;
	pscnv_chan_unref(chan);
fail_chan:
	kfree(ctx);
fail_ctx:
	return NULL;
}

/* destroy the specified GPU context object. */
void gdev_raw_ctx_free(struct gdev_ctx *ctx)
{
	struct gdev_vas *vas = ctx->vas; 
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_chan *chan = ctx->pctx;
	struct pscnv_bo *fence_bo = ctx->fence.bo;

	kunmap(fence_bo->pages[0]);
	pscnv_vspace_unmap(vspace, ctx->fence.addr);
	pscnv_mem_free((struct pscnv_bo *)ctx->fence.bo);
	vunmap(ctx->fifo.pb_map);
	pscnv_vspace_unmap(vspace, ctx->fifo.pb_base);
	pscnv_mem_free((struct pscnv_bo *)ctx->fifo.pb_bo);
	vunmap(ctx->fifo.ib_map);
	pscnv_vspace_unmap(vspace, ctx->fifo.ib_base);
	pscnv_mem_free((struct pscnv_bo *)ctx->fifo.ib_bo);

	chan->filp = NULL;
	pscnv_chan_unref(chan);

	kfree(ctx);
}

/* allocate a new memory object. */
static inline struct gdev_mem *__gdev_raw_mem_alloc
(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map, 
 uint32_t flags)
{
	struct gdev_mem *mem;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo;
	struct pscnv_mm_node *mm;
	uint64_t raw_size = *size;

	if (!(mem = kzalloc(sizeof(*mem), GFP_KERNEL)))
		goto fail_mem;

	if (!(bo = pscnv_mem_alloc(drm, raw_size, flags, 0, 0)))
		goto fail_bo;

	if (pscnv_vspace_map(vspace, bo, GDEV_VAS_USER_START, 
						 GDEV_VAS_USER_END, 0, &mm))
		goto fail_map;

	/* address, size, and map. */
	*addr = mm->start;
	*size = bo->size;
	if (flags & PSCNV_GEM_SYSRAM_SNOOP) {
		if (bo->size > PAGE_SIZE)
			*map = vmap(bo->pages, bo->size >> PAGE_SHIFT, 0, PAGE_KERNEL);
		else
			*map = kmap(bo->pages[0]);
	}
	else
		*map = NULL;

	/* private data. */
	mem->bo = (void *) bo;

	return mem;

fail_map:
	GDEV_PRINT("Failed to map VAS.\n");
	pscnv_mem_free(bo);
fail_bo:
	GDEV_PRINT("Failed to allocate PSCNV buffer object.\n");
	kfree(mem);
fail_mem:
	return NULL;
}

/* allocate a new device memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc
(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map)
{
	return __gdev_raw_mem_alloc(vas, addr, size, map, PSCNV_GEM_VRAM_SMALL);
}

/* allocate a new host DMA memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc_dma
(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map)
{
	return __gdev_raw_mem_alloc(vas, addr, size, map, PSCNV_GEM_SYSRAM_SNOOP);
}

/* free the specified memory object. */
void gdev_raw_mem_free(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo = mem->bo;

	if (mem->map) {
		if (bo->size > PAGE_SIZE)
			vunmap(mem->map);
		else
			kunmap(mem->map);
	}
	pscnv_vspace_unmap(vspace, mem->addr);
	pscnv_mem_free(bo);
	kfree(mem);
}

/* create a new memory object sharing memory space with @mem. */
struct gdev_mem *gdev_raw_mem_share
(struct gdev_vas *vas, struct gdev_mem *mem, uint64_t *addr, uint64_t *size, 
 void **map)
{
	struct pscnv_vspace *vs = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	struct pscnv_mm_node *mm;
	struct gdev_mem *new;

	if (!(new = kzalloc(sizeof(*new), GFP_KERNEL)))
		goto fail_mem;

	if (pscnv_vspace_map(vs, bo, GDEV_VAS_USER_START, GDEV_VAS_USER_END, 0, 
						 &mm))
		goto fail_map;

	/* address, size, and map. */
	*addr = mm->start;
	*size = bo->size;
	if (bo->flags & PSCNV_GEM_SYSRAM_SNOOP) {
		if (bo->size > PAGE_SIZE)
			*map = vmap(bo->pages, bo->size >> PAGE_SHIFT, 0, PAGE_KERNEL);
		else
			*map = kmap(bo->pages[0]);
	}
	else
		*map = NULL;

	/* private data. */
	new->bo = (void *) bo;

	return new;

fail_map:
	kfree(new);
fail_mem:
	return NULL;
}

/* destroy the memory object by just unsharing memory space. */
void gdev_raw_mem_unshare(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct pscnv_vspace *vspace = vas->pvas;

	pscnv_vspace_unmap(vspace, mem->addr);
	kfree(mem);
}
