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
#include "gdev_drv.h"
#include "gdev_list.h"
#include "gdev_nvidia.h"
#include "gdev_sched.h"
#include "gdev_interface.h"

/* query device-specific information. */
int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *res)
{
	struct drm_device *drm = (struct drm_device *) gdev->priv;

	switch (type) {
	case GDEV_NVIDIA_QUERY_MP_COUNT:
		return gdev_drv_getparam(drv, GDEV_DRV_GETPARAM_MP_COUNT, res);
	case GDEV_QUERY_DEVICE_MEM_SIZE:
		return gdev_drv_getparam(drv, GDEV_DRV_GETPARAM_FB_SIZE, res);
	case GDEV_QUERY_DMA_MEM_SIZE:
		return gdev_drv_getparam(drv, GDEV_DRV_GETPARAM_AGP_SIZE, res);
	case GDEV_QUERY_CHIPSET:
		return gdev_drv_getparam(drv, GDEV_DRV_GETPARAM_CHIPSET_ID, res);
	default:
		return -EINVAL;
	}
}

/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_raw_dev_open(int minor)
{
	struct gdev_device *gdev = &gdev_vds[minor]; /* virutal device */

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
	struct gdev_drv_vspace vspace;
	struct drm_device *drm = (struct drm_device *) gdev->priv;

	if (!(vas = kzalloc(sizeof(*vas), GFP_KERNEL)))
		goto fail_vas;

	/* call the device driver specific function. */
	if (gdev_drv_vspace_alloc(drm, size, &vspace)))
		goto fail_vspace;

	vas->pvas = vspace.priv; /* driver private object. */

	return vas;

fail_vspace:
	kfree(vas);
fail_vas:
	return NULL;
}

/* free the specified virtual address space object. */
void gdev_raw_vas_free(struct gdev_vas *vas)
{
	struct gdev_drv_vspace vspace;

	vspace.priv = vas->pvas;
	pscnv_vspace_unref(&vspace);
	kfree(vas);
}

/* create a new GPU context object. */
struct gdev_ctx *gdev_raw_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	struct gdev_drv_chan chan;
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo fbo, nbo;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint32_t flags;

	if (!(ctx = kzalloc(sizeof(*ctx), GFP_KERNEL)))
		goto fail_ctx;

	vspace.priv = vas->pvas; 
	if (gdev_drv_chan_alloc(drm, &vspace, &chan))
		goto fail_chan;

	ctx->cid = chan.cid;
	ctx->pctx = chan.priv; /* driver private data. */
	ctx->vas = vas;

	/* command FIFO. */
	ctx->fifo.regs = chan.regs;
	ctx->fifo.ib_bo = chan.ib_bo;
	ctx->fifo.ib_map = chan.ib_map;
	ctx->fifo.ib_order = chan.ib_order;
	ctx->fifo.ib_base = chan->ib_base;
	ctx->fifo.ib_mask = chan->ib_mask;
	ctx->fifo.ib_put = 0;
	ctx->fifo.ib_get = 0;
	ctx->fifo.pb_bo = chan->pb_bo;
	ctx->fifo.pb_map = chan->pb_map;
	ctx->fifo.pb_order = chan->pb_order;
	ctx->fifo.pb_base = chan->pb_base;
	ctx->fifo.pb_mask = chan->pb_mask;
	ctx->fifo.pb_size = chan->pb_size;
	ctx->fifo.pb_pos = 0;
	ctx->fifo.pb_put = 0;
	ctx->fifo.pb_get = 0;

	/* fence buffer. */
	flags = GDEV_DRV_BO_SYSRAM | GDEV_DRV_BO_VSPACE | GDEV_DRV_BO_MAPPABLE;
	if (gdev_drv_bo_alloc(drm, GDEV_FENCE_BUF_SIZE, flags, &vspace, &fbo))
		goto fail_fence_alloc;
	ctx->fence.bo = fbo.priv;
	ctx->fence.addr = fbo.addr;
	ctx->fence.map = fbo.map;
	ctx->fence.seq = 0;

	/* notify buffer. */
	flags = GDEV_DRV_BO_VRAM | GDEV_DRV_BO_VSPACE;
	if (gdev_drv_bo_alloc(drm, 8 /* 64 bits */, flags, &vspace, &nbo))
		goto fail_notify_alloc;
	ctx->notify.bo = nbo.priv;
	ctx->notify.addr = nbo.addr;

	return ctx;
	
fail_notify_alloc:
	gdev_drv_bo_free(&vspace, &fbo);
fail_fence_alloc:
	gdev_drv_chan_free(&vspace, &chan);
fail_chan:
	kfree(ctx);
fail_ctx:
	return NULL;
}

/* destroy the specified GPU context object. */
void gdev_raw_ctx_free(struct gdev_ctx *ctx)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_chan chan;
	struct gdev_drv_bo fbo, nbo;
	struct gdev_vas *vas = ctx->vas; 

	vspace.priv = vas->pvas;

	nmem.priv = ctx->notify.bo;
	nmem.addr = ctx->notify.addr;
	gdev_drv_bo_free(&vspace, &nbo);

	fmem.priv = ctx->fence.bo;
	fmem.addr = ctx->fence.addr;
	fmem.map = ctx->fence.map;
	gdev_drv_bo_free(&vspace, &fbo);

	chan.priv = ctx->pctx;
	chan.ib_bo = ctx->fifo.ib_bo;
	chan.ib_base = ctx->fifo.ib_base;
	chan.ib_map = ctx->fifo.ib_map;
	chan.pb_bo = ctx->fifo.pb_bo;
	chan.pb_base = ctx->fifo.pb_base;
	chan.pb_map = ctx->fifo.pb_map;
	gdev_drv_chan_free(&vspace, &chan);

	kfree(ctx);
}

/* allocate a new memory object. */
static inline struct gdev_mem *__gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map, uint32_t flags)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_mem *mem;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;

	GDEV_DPRINT("Allocating memory of 0x%llx bytes\n", *size);

	if (!(mem = kzalloc(sizeof(*mem), GFP_KERNEL)))
		goto fail_mem;

	vspace.priv = vas->pvas;
	if (gdev_drv_bo_alloc(drv, *size, flags, &vspace, &bo))
		goto fail_bo_alloc;
	*addr = bo.addr;
	*size = bo.size;
	*map = bo.map;
	mem->bo = bo.priv;
	
	return mem;

fail_mem_alloc:
	GDEV_PRINT("Failed to allocate driver buffer object\n");
	kfree(mem);
fail_mem:
	GDEV_PRINT("Failed to allocate memory object\n");
	return NULL;
}

/* allocate a new device memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map)
{
	uint32_t flags = GDEV_DRV_BO_VRAM | GDEV_DRV_BO_VSPACE;

	if (*size <= GDEV_MEM_MAPPABLE_LIMIT)
		flags |= GDEV_DRV_BO_MAPPABLE;

	return __gdev_raw_mem_alloc(vas, addr, size, map, flags);
}

/* allocate a new host DMA memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc_dma(struct gdev_vas *vas, uint64_t *addr, uint64_t *size, void **map)
{
	uint32_t flags = GDEV_DRV_BO_SYSRAM | GDEV_DRV_BO_VSPACE | GDEV_DRV_BO_MAPPABLE; /* dma host memory is always mapped to user buffers. */
	return __gdev_raw_mem_alloc(vas, addr, size, map, flags);
}

/* free the specified memory object. */
void gdev_raw_mem_free(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;

	vspace.priv = vas->pvas;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size;
	bo.map = mem->map;
	if (gdev_drv_bo_free(&vspace, &bo))
		GDEV_PRINT("Failed to free driver buffer object\n");
	kfree(mem);
}

/* allocate a reserved swap memory object. size may be aligned. */
struct gdev_mem *gdev_raw_swap_alloc(struct gdev_device *gdev, uint64_t size)
{
	struct gdev_mem *mem;
	struct gdev_drv_bo bo;
	struct gdev_drv_vspace vspace;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint32_t flags = GDEV_DRV_BO_VRAM;

	if (size == 0)
		goto fail_size;

	if (!(mem = kzalloc(sizeof(*mem), GFP_KERNEL)))
		goto fail_mem;

	vspace.priv = NULL;
	if (gdev_drv_bo_alloc(drm, size, flags, &vspace, &bo))
		goto fail_bo_alloc;

	mem->bo = bo.priv;
	mem->size = bo.size;

	return mem;

fail_bo_alloc:
	GDEV_PRINT("Failed to allocate driver buffer object\n");
	kfree(mem);
fail_mem:
fail_size:
	GDEV_PRINT("Failed to allocate swap memory object\n");
	return NULL;
}

/* free the specified swap memory object. */
void gdev_raw_swap_free(struct gdev_mem *mem)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;

	if (mem) {
		vspace.priv = NULL;
		bo.priv = mem->bo;
		bo.addr = 0;
		bo.map = NULL;
		gdev_drv_bo_free(&vspace, &bo);
		kfree(mem);
	}
}

/* create a new memory object sharing memory space with @mem. */
struct gdev_mem *gdev_raw_mem_share(struct gdev_vas *vas, struct gdev_mem *mem, uint64_t *addr, uint64_t *size, void **map)
{
	struct pscnv_vspace *vs = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	struct pscnv_mm_node *mm;
	struct gdev_mem *new;

	if (!(new = kzalloc(sizeof(*new), GFP_KERNEL)))
		goto fail_mem;

	if (pscnv_vspace_map(vs, bo, GDEV_VAS_USER_START, GDEV_VAS_USER_END, 0,&mm))
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

	GDEV_DPRINT("Shared memory of 0x%llx bytes at 0x%llx\n", *size, *addr);

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

/* map device memory to host DMA memory. */
void *gdev_raw_mem_map(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	struct pscnv_bo *bo = mem->bo;
	unsigned long bar1_start = pci_resource_start(drm->pdev, 1);
	void *map;

	if (dev_priv->vm->map_user(bo))
		goto fail_map_user;
	if (!(map = ioremap(bar1_start + bo->map1->start, bo->size)))
		goto fail_ioremap;

	bo->flags |= PSCNV_GEM_MAPPABLE;

	return map;

fail_ioremap:
	GDEV_PRINT("Failed to map PCI BAR1\n");
	pscnv_vspace_unmap_node(bo->map1);
fail_map_user:
	GDEV_PRINT("Failed to map host and device memory\n");

	return NULL;
}

/* unmap device memory from host DMA memory. */
void gdev_raw_mem_unmap(struct gdev_mem *mem, void *map)
{
	struct pscnv_bo *bo = mem->bo;

	iounmap(map);
	pscnv_vspace_unmap_node(bo->map1);
	bo->flags &= ~PSCNV_GEM_MAPPABLE;
}

/* get physical bus address. */
uint64_t gdev_raw_mem_phys_getaddr(struct gdev_mem *mem, uint64_t offset)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	int page = offset / PAGE_SIZE;
	uint32_t x = offset - page * PAGE_SIZE;
	uint32_t flags = bo->flags;

	if (flags & PSCNV_GEM_MAPPABLE) {
		if (flags & PSCNV_GEM_SYSRAM_SNOOP)
			return bo->dmapages[page] + x;
		else
			return pci_resource_start(drm->pdev, 1) + bo->map1->start + x;
	}
	else {
		return dev_priv->vm->phys_getaddr(vspace, bo, mem->addr + offset);
	}
}

uint32_t gdev_raw_read32(struct gdev_mem *mem, uint64_t addr)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	uint64_t offset = addr - mem->addr;
	uint32_t val;

	if (mem->map) {
		val = ioread32_native(mem->map + offset);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->read32(vspace, bo, addr, &val);
		mutex_unlock(&vspace->lock);
	}

	return val;
}

void gdev_raw_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	uint64_t offset = addr - mem->addr;

	if (mem->map) {
		iowrite32_native(val, mem->map + offset);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->write32(vspace, bo, addr, val);
		mutex_unlock(&vspace->lock);
	}
}

int gdev_raw_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	uint64_t offset = addr - mem->addr;

	if (mem->map) {
		memcpy_fromio(buf, mem->map + offset, size);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->read(vspace, bo, addr, buf, size);
		mutex_unlock(&vspace->lock);
	}
		
	return 0;
}

int gdev_raw_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct pscnv_vspace *vspace = vas->pvas;
	struct pscnv_bo *bo = mem->bo;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	uint64_t offset = addr - mem->addr;

	if (mem->map) {
		memcpy_toio(mem->map + offset, buf, size);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->write(vspace, bo, addr, buf, size);
		mutex_unlock(&vspace->lock);
	}

	return 0;
}
