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

#include "gdev_api.h"
#include "gdev_device.h"
#include "gdev_drv.h"
#include "gdev_interface.h"
#include "gdev_list.h"
#include "gdev_nvidia.h"
#include "gdev_nvidia_fifo.h"
#include "gdev_nvidia_nve4.h"
#include "gdev_sched.h"

#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
struct gdev_drv_nvidia_pdata {
	struct drm_device *drm;
};
#endif

/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_raw_dev_open(int minor)
{
	struct gdev_device *gdev = &gdev_vds[minor]; /* virutal device */

	/* fix this */
#if 1
	struct gdev_device *phys = gdev->parent;
	if(phys){
retry:
	    gdev_lock(&phys->global_lock);
	    if(phys->users > GDEV_CONTEXT_LIMIT){
		gdev_unlock(&phys->global_lock);
		schedule_timeout(5);
		goto retry;
	    }
	    phys->users++; 
	    gdev_unlock(&phys->global_lock);
	}
#endif
	gdev->users++;

	return gdev;
}

/* close the specified Gdev object. */
void gdev_raw_dev_close(struct gdev_device *gdev)
{
#if 1
	struct gdev_device *phys = gdev->parent;
	if(phys){
	    phys->users--;
	}
#endif
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
	if (gdev_drv_vspace_alloc(drm, size, &vspace))
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
	gdev_drv_vspace_free(&vspace);
	kfree(vas);
}

/* create a new GPU context object. */
struct gdev_ctx *gdev_raw_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	struct gdev_drv_chan chan;
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo fbo, nbo, dbo;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint32_t flags;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
	struct gdev_drv_nvidia_pdata *pdata;
	void *m2mf;
	uint32_t m2mf_class = 0;
#if 0 /* un-necessary */
	void *comp;
	uint32_t comp_class = 0;
#endif
#endif


	if (!(ctx = kzalloc(sizeof(*ctx), GFP_KERNEL)))
		goto fail_ctx;

	vspace.priv = vas->pvas;

	if (gdev_drv_chan_alloc(drm, &vspace, &chan))
		goto fail_chan;

	ctx->cid = chan.cid & 0xffff;
	ctx->pctx = chan.priv; /* driver private data. */
	ctx->vas = vas;

	/* command FIFO. */
	ctx->fifo.regs = chan.regs;
	ctx->fifo.ib_bo = chan.ib_bo;
	ctx->fifo.ib_map = chan.ib_map;
	ctx->fifo.ib_order = chan.ib_order;
	ctx->fifo.ib_base = chan.ib_base;
	ctx->fifo.ib_mask = chan.ib_mask;
	ctx->fifo.ib_put = 0;
	ctx->fifo.ib_get = 0;
	ctx->fifo.pb_bo = chan.pb_bo;
	ctx->fifo.pb_map = chan.pb_map;
	ctx->fifo.pb_order = chan.pb_order;
	ctx->fifo.pb_base = chan.pb_base;
	ctx->fifo.pb_mask = chan.pb_mask;
	ctx->fifo.pb_size = chan.pb_size;
	ctx->fifo.pb_pos = 0;
	ctx->fifo.pb_put = 0;
	ctx->fifo.pb_get = 0;
	ctx->fifo.push = gdev_fifo_push;
	ctx->fifo.update_get = gdev_fifo_update_get;

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
	
	/* compute desc buffer.
	 * In fact, it must be created for each kernel launch.
	 * need fix.
	 */
	if ((gdev->chipset & 0xf0) >= 0xe0){
	    flags = GDEV_DRV_BO_SYSRAM | GDEV_DRV_BO_VSPACE | GDEV_DRV_BO_MAPPABLE;
	    if (gdev_drv_bo_alloc(drm, sizeof(struct gdev_nve4_compute_desc), flags, &vspace, &dbo)){
		goto fail_desc_alloc;
	    }
	    ctx->desc.bo = dbo.priv;
	    ctx->desc.addr = dbo.addr;
	    ctx->desc.map = dbo.map;
	    memset(dbo.map, 0,sizeof(struct gdev_nve4_compute_desc));
	}else{
	    ctx->desc.bo = NULL;
	}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
	if (!(pdata = kzalloc(sizeof(*pdata), GFP_KERNEL)))
		goto fail_ctx_objects;

	pdata->drm = drm;

        /* allocating PGRAPH context for M2MF */
	if ((gdev->chipset & 0xf0) < 0xc0)
		m2mf_class = 0x5039;
	else if ((gdev->chipset & 0xf0) < 0xe0)
		m2mf_class = 0x9039;
	else
		m2mf_class = 0xa040;
	if (gdev_drv_subch_alloc(drm, ctx->pctx, 0xbeef323f, m2mf_class, &m2mf))
		goto fail_m2mf;
#if 0 /* un-necessary */
	/* allocating PGRAPH context for COMPUTE */
	if ((gdev->chipset & 0xf0) < 0xc0)
		comp_class = 0x50c0;
	else if ((gdev->chipset & 0xf0) < 0xe0)
		comp_class = 0x90c0;
	else
		comp_class = 0xa0c0;
	if (gdev_drv_subch_alloc(drm, ctx->pctx, 0xbeef90c0, comp_class, &comp))
		goto fail_comp;
#endif

	ctx->pdata = (void *)pdata;
#endif

	return ctx;
	
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
#if 0 /* un-necessary */
fail_comp:
	gdev_drv_subch_free(drm, ctx->pctx, 0xbeef323f);
#endif
fail_m2mf:
	kfree(pdata);
fail_ctx_objects:
	gdev_drv_bo_free(&vspace, &nbo);
#endif
fail_desc_alloc:
	gdev_drv_bo_free(&vspace, &dbo);
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
	struct gdev_drv_bo fbo, nbo, dbo;
	struct gdev_vas *vas = ctx->vas; 
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
	struct gdev_drv_nvidia_pdata *pdata = (struct gdev_drv_nvidia_pdata *)ctx->pdata;
#endif

	vspace.priv = vas->pvas;

	nbo.priv = ctx->notify.bo;
	nbo.addr = ctx->notify.addr;
	gdev_drv_bo_free(&vspace, &nbo);

	fbo.priv = ctx->fence.bo;
	fbo.addr = ctx->fence.addr;
	fbo.map = ctx->fence.map;
	gdev_drv_bo_free(&vspace, &fbo);

	/* compute desc buffer is allocated only when a target chipset
	 * is NVE4 or later. Some chipset like NVC0 doesn't have it.
	 */
	if (ctx->desc.bo) {
		dbo.priv = ctx->desc.bo;
		dbo.addr = ctx->desc.addr;
		gdev_drv_bo_free(&vspace, &dbo);
	}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
#if 0 /* un-necessary */
	gdev_drv_subch_free(pdata->drm, ctx->pctx, 0xbeef90c0);
#endif
	gdev_drv_subch_free(pdata->drm, ctx->pctx, 0xbeef323f);
	kfree(pdata);
#endif

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
static inline struct gdev_mem *__gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t size, uint32_t flags)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_mem *mem;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;

	GDEV_DPRINT("Allocating memory of 0x%llx bytes\n", size);

	if (!(mem = kzalloc(sizeof(*mem), GFP_KERNEL)))
		goto fail_mem;

	vspace.priv = vas->pvas;
	if (gdev_drv_bo_alloc(drm, size, flags, &vspace, &bo))
		goto fail_bo_alloc;
	mem->addr = bo.addr;
	mem->size = bo.size;
	mem->map = bo.map;
	mem->bo = bo.priv;
	mem->pdata = (void *)drm;
	
	return mem;

fail_bo_alloc:
	GDEV_PRINT("Failed to allocate driver buffer object\n");
	kfree(mem);
fail_mem:
	GDEV_PRINT("Failed to allocate memory object\n");
	return NULL;
}

/* allocate a new device memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t size)
{
	uint32_t flags = GDEV_DRV_BO_VRAM | GDEV_DRV_BO_VSPACE;

	if (size <= GDEV_MEM_MAPPABLE_LIMIT)
		flags |= GDEV_DRV_BO_MAPPABLE;

	return __gdev_raw_mem_alloc(vas, size, flags);
}

/* allocate a new host DMA memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc_dma(struct gdev_vas *vas, uint64_t size)
{
	uint32_t flags = GDEV_DRV_BO_SYSRAM | GDEV_DRV_BO_VSPACE | GDEV_DRV_BO_MAPPABLE; /* dma host memory is always mapped to user buffers. */
	return __gdev_raw_mem_alloc(vas, size, flags);
}

/* free the specified memory object. */
void gdev_raw_mem_free(struct gdev_mem *mem)
{
	struct gdev_vas *vas = mem->vas;
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;

	vspace.priv = vas->pvas;
	vspace.drm = mem->pdata;
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
	mem->addr = 0;
	mem->size = bo.size;
	mem->map = NULL;
	mem->pdata = (void *)drm;

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
		vspace.priv = NULL; /* indicate that bo doensn't have vspace. */
		vspace.drm = mem->pdata;
		bo.priv = mem->bo;
		bo.addr = mem->addr; /* not really used. */
		bo.size = mem->size; /* not really used. */
		bo.map = mem->map; /* not really used. */
		gdev_drv_bo_free(&vspace, &bo);
		kfree(mem);
	}
}

/* create a new memory object sharing memory space with @mem. */
struct gdev_mem *gdev_raw_mem_share(struct gdev_vas *vas, struct gdev_mem *mem)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_mem *new;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *)gdev->priv;

	if (!(new = kzalloc(sizeof(*new), GFP_KERNEL)))
		goto fail_mem;

	vspace.priv = vas->pvas;
	vspace.drm = mem->pdata;
	bo.priv = mem->bo;
	bo.addr = 0; /* will be obtained. */
	bo.size = 0; /* will be obtained. */
	bo.map = NULL; /* will be obtained. */

	/* bind a virtual address in @vspace to memory space in @bo. */
	if (gdev_drv_bo_bind(drm, &vspace, &bo))
		goto fail_bind;

	/* address, size, and map. */
	new->addr = bo.addr;
	new->size = bo.size;
	new->map = bo.map;
	new->bo = (void *)bo.priv; /* private driver object. */
	new->pdata = (void *)drm;

	GDEV_DPRINT("Shared memory of 0x%llx bytes at 0x%llx\n", bo.size, bo.addr);

	return new;

fail_bind:
	kfree(new);
fail_mem:
	return NULL;
}

/* destroy the memory object by just unsharing memory space. */
void gdev_raw_mem_unshare(struct gdev_mem *mem)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;

	vspace.priv = vas->pvas;
	vspace.drm = mem->pdata;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map; /* not really used. */

	gdev_drv_bo_unbind(&vspace, &bo);
	kfree(mem);
}

/* map device memory to host DMA memory. */
void *gdev_raw_mem_map(struct gdev_mem *mem)
{
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *)gdev->priv;

	if (mem->map)
		return mem->map;

	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size;
	bo.map = NULL; /* will be obtained. */
	if (gdev_drv_bo_map(drm, &bo))
		goto fail_map;

	return bo.map;

fail_map:
	GDEV_PRINT("Failed to map host and device memory\n");
	return NULL;
}

/* unmap device memory from host DMA memory. */
void gdev_raw_mem_unmap(struct gdev_mem *mem, void *map)
{
	struct gdev_drv_bo bo;

	bo.priv = mem->bo;
	bo.addr = mem->addr; /* not really used. */
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map;
	
	gdev_drv_bo_unmap(&bo);
}

uint32_t gdev_raw_read32(struct gdev_mem *mem, uint64_t addr)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint64_t offset = addr - mem->addr;
	uint32_t val;

	vspace.priv = vas->pvas;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map;

	gdev_drv_read32(drm, &vspace, &bo, offset, &val);

	return val;
}

void gdev_raw_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint64_t offset = addr - mem->addr;

	vspace.priv = vas->pvas;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map;

	gdev_drv_write32(drm, &vspace, &bo, offset, val);
}

int gdev_raw_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint64_t offset = addr - mem->addr;

	vspace.priv = vas->pvas;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map;

	gdev_drv_read(drm, &vspace, &bo, offset, size, buf);
		
	return 0;
}

int gdev_raw_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint64_t offset = addr - mem->addr;

	vspace.priv = vas->pvas;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map;

	gdev_drv_write(drm, &vspace, &bo, offset, size, buf);

	return 0;
}

/* get physical bus address. */
uint64_t gdev_raw_mem_phys_getaddr(struct gdev_mem *mem, uint64_t offset)
{
	struct gdev_drv_vspace vspace;
	struct gdev_drv_bo bo;
	struct gdev_vas *vas = mem->vas;
	struct gdev_device *gdev = vas->gdev;
	struct drm_device *drm = (struct drm_device *) gdev->priv;
	uint64_t phys;

	vspace.priv = vas->pvas;
	bo.priv = mem->bo;
	bo.addr = mem->addr;
	bo.size = mem->size; /* not really used. */
	bo.map = mem->map;

	gdev_drv_getaddr(drm, &vspace, &bo, offset, &phys);

	return phys;
}

/* query device-specific information. */
int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *res)
{
	struct drm_device *drm = (struct drm_device *) gdev->priv;

	switch (type) {
	case GDEV_NVIDIA_QUERY_MP_COUNT:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_MP_COUNT, res);
	case GDEV_QUERY_DEVICE_MEM_SIZE:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_FB_SIZE, res);
	case GDEV_QUERY_DMA_MEM_SIZE:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_AGP_SIZE, res);
	case GDEV_QUERY_CHIPSET:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_CHIPSET_ID, res);
	case GDEV_QUERY_BUS_TYPE:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_BUS_TYPE, res);
	case GDEV_QUERY_AGP_SIZE:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_AGP_SIZE, res);
	case GDEV_QUERY_PCI_VENDOR:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_PCI_VENDOR, res);
	case GDEV_QUERY_PCI_DEVICE:
		return gdev_drv_getparam(drm, GDEV_DRV_GETPARAM_PCI_DEVICE, res);
	default:
		return -EINVAL;
	}
}
