#include <linux/module.h>
#include "gdev_interface.h"
#include "nouveau_drv.h"
#include "nvc0_graph.h"

#define VS_START 0x20000000
#define VS_END (1ull << 40)

extern int nouveau_device_count;
extern struct drm_device **nouveau_drm;
extern void (*nouveau_callback_notify)(int subc, uint32_t data);

int gdev_drv_vspace_alloc(struct drm_device *drm, uint64_t size, struct gdev_drv_vspace *drv_vspace)
{
	struct nouveau_channel *chan;

	if (nouveau_channel_alloc(drm, &chan, NULL, 0xbeef0201, 0xbeef0202)) {
		printk("Failed to allocate nouveau channel\n");
		return -ENOMEM;
	}

	drv_vspace->priv = (void *)chan;
	
	return 0;
}
EXPORT_SYMBOL(gdev_drv_vspace_alloc);

int gdev_drv_vspace_free(struct gdev_drv_vspace *drv_vspace)
{
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;

	nouveau_channel_put(&chan);
	
	return 0;
}
EXPORT_SYMBOL(gdev_drv_vspace_free);

int gdev_drv_chan_alloc(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_chan *drv_chan)
{
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;
	struct nouveau_bo *ib_bo, *pb_bo;
	uint32_t cid;
	volatile uint32_t *regs;
	uint32_t *ib_map, *pb_map;
	uint32_t ib_order, pb_order;
	uint64_t ib_base, pb_base;
	uint32_t ib_mask, pb_mask;
	uint32_t pb_size;
	int ret;
	
	/* channel ID. */
	cid = chan->id;

	/* FIFO push buffer setup. */
	pb_order = 15; /* it's hardcoded. pscnv uses 20, nouveau uses 15. */
	pb_bo = chan->pushbuf_bo;
	pb_base = chan->pushbuf_vma.offset;
	pb_map = chan->pushbuf_bo->kmap.virtual;
	pb_mask = (1 << pb_order) - 1;
	pb_size = (1 << pb_order);
	if (chan->pushbuf_bo->bo.mem.size / 2 != pb_size)
		printk("Pushbuf size mismatched!\n");

	/* FIFO indirect buffer setup. */
	ib_order = 12; /* it's hardcoded. pscnv uses 9, nouveau uses 12*/
	ib_bo = NULL;
	ib_base = pb_base + pb_size;
	ib_map = (void *)((unsigned long)pb_bo->kmap.virtual + pb_size);
	ib_mask = (1 << ib_order) - 1;

	/* FIFO init: it has already been done in gdev_vas_new(). */

	switch (dev_priv->chipset & 0xf0) {
	case 0xc0:
		/* FIFO command queue registers. */
		regs = chan->user;
		/* PCOPY engines. */
		ret = dev_priv->eng[NVOBJ_ENGINE_COPY0]->context_new(chan, NVOBJ_ENGINE_COPY0);
		if (ret)
			goto fail_pcopy0;
		ret = dev_priv->eng[NVOBJ_ENGINE_COPY1]->context_new(chan, NVOBJ_ENGINE_COPY1);
		if (ret)
			goto fail_pcopy1;
		break;
	default:
		ret = -EINVAL;
		goto fail_fifo_reg;
	}

	drv_chan->priv = chan;
	drv_chan->cid = cid;
	drv_chan->regs = regs;
	drv_chan->ib_bo = ib_bo;
	drv_chan->ib_map = ib_map;
	drv_chan->ib_order = ib_order;
	drv_chan->ib_base = ib_base;
	drv_chan->ib_mask = ib_mask;
	drv_chan->pb_bo = pb_bo;
	drv_chan->pb_map = pb_map;
	drv_chan->pb_order = pb_order;
	drv_chan->pb_base = pb_base;
	drv_chan->pb_mask = pb_mask;
	drv_chan->pb_size = pb_size;

	return 0;

fail_pcopy1:
	dev_priv->eng[NVOBJ_ENGINE_COPY0]->context_del(chan, NVOBJ_ENGINE_COPY0);
fail_fifo_reg:
fail_pcopy0:
	return ret;
}
EXPORT_SYMBOL(gdev_drv_chan_alloc);

int gdev_drv_chan_free(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_chan *drv_chan)
{
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;
	struct drm_nouveau_private *dev_priv = chan->dev->dev_private;

	dev_priv->eng[NVOBJ_ENGINE_COPY1]->context_del(chan, NVOBJ_ENGINE_COPY1);
	dev_priv->eng[NVOBJ_ENGINE_COPY0]->context_del(chan, NVOBJ_ENGINE_COPY0);

	return 0;
}
EXPORT_SYMBOL(gdev_drv_chan_free);

int gdev_drv_bo_alloc(struct drm_device *drm, uint64_t size, uint32_t drv_flags, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;
	struct nouveau_bo *bo;
	struct nouveau_vma *vma;
	uint32_t flags = 0;
	int ret;

	/* set memory type. */
	if (drv_flags & GDEV_DRV_BO_VRAM) {
		flags |= TTM_PL_FLAG_VRAM;
	}
	if (drv_flags & GDEV_DRV_BO_SYSRAM) {
		flags |= TTM_PL_FLAG_TT;
	}

	ret = nouveau_bo_new(drm, size, 0, flags, 0, 0, &bo);
	if (ret)
		goto fail_bo_new;

	if (drv_flags & GDEV_DRV_BO_MAPPABLE) {
		ret = nouveau_bo_map(bo);
		if (ret)
			goto fail_bo_map;
	}
	else
		bo->kmap.virtual = NULL;

	/* allocate virtual address space, if requested. */
	if (drv_flags & GDEV_DRV_BO_VSPACE) {
		if (dev_priv->card_type >= NV_50) {
			vma = kzalloc(sizeof(*vma), GFP_KERNEL);
			if (!vma) {
				ret = -ENOMEM;
				goto fail_vma_alloc;
			}

			ret = nouveau_bo_vma_add(bo, chan->vm, vma);
			if (ret)
				goto fail_vma_add;

			drv_bo->addr = vma->offset;
		}
		else /* non-supported cards. */
			drv_bo->addr = 0;
	}
	else
		drv_bo->addr = 0;

	/* address, size, and map. */
	if (bo->kmap.virtual) 
		drv_bo->map = bo->kmap.virtual;
	else
		drv_bo->map = NULL;
	drv_bo->size = bo->bo.mem.size;
	drv_bo->priv = bo;

	return 0;

fail_vma_add:
	kfree(vma);
fail_vma_alloc:
	nouveau_bo_unmap(bo);
fail_bo_map:
	nouveau_bo_ref(NULL, &bo);
fail_bo_new:
	return ret;

}
EXPORT_SYMBOL(gdev_drv_bo_alloc);

int gdev_drv_bo_free(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;
	struct nouveau_bo *bo = (struct nouveau_bo *)drv_bo->priv;
	struct nouveau_vma *vma;
	uint64_t addr = drv_bo->addr;
	void *map = drv_bo->map;

	if (map && bo->kmap.bo) /* dirty validation.. */
		nouveau_bo_unmap(bo);

	if (addr) {
		vma = nouveau_bo_vma_find(bo, chan->vm);
		if (vma) {
			nouveau_bo_vma_del(bo, vma);
			kfree(vma);
		}
		else {
			return -ENOENT;
		}
	}

	nouveau_bo_ref(NULL, &bo);

	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_free);

int gdev_drv_bo_bind(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;
	struct nouveau_bo *bo = (struct nouveau_bo *)drv_bo->priv;
	struct nouveau_vma *vma;
	int ret;

	/* allocate virtual address space, if requested. */
	if (dev_priv->card_type >= NV_50) {
		vma = kzalloc(sizeof(*vma), GFP_KERNEL);
		if (!vma) {
			ret = -ENOMEM;
			goto fail_vma_alloc;
		}
		
		ret = nouveau_bo_vma_add(bo, chan->vm, vma);
		if (ret)
			goto fail_vma_add;
		
		drv_bo->addr = vma->offset;
	}
	else /* non-supported cards. */
		drv_bo->addr = 0;

	drv_bo->map = bo->kmap.virtual; /* could be NULL. */
	drv_bo->size = bo->bo.mem.size;

	return 0;

fail_vma_add:
	kfree(vma);
fail_vma_alloc:
	return ret;
}
EXPORT_SYMBOL(gdev_drv_bo_bind);

int gdev_drv_bo_unbind(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct nouveau_channel *chan = (struct nouveau_channel *)drv_vspace->priv;
	struct nouveau_bo *bo = (struct nouveau_bo *)drv_bo->priv;
	struct nouveau_vma *vma;

	vma = nouveau_bo_vma_find(bo, chan->vm);
	if (vma) {
		nouveau_bo_vma_del(bo, vma);
		kfree(vma);
	}
	else
		return -ENOENT;
	
	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_unbind);

int gdev_drv_bo_map(struct drm_device *drm, struct gdev_drv_bo *drv_bo)
{
	struct nouveau_bo *bo = (struct nouveau_bo *)drv_bo->priv;
	int ret;

	ret = nouveau_bo_map(bo);
	if (ret)
		return ret;

	drv_bo->map = bo->kmap.virtual;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_map);

int gdev_drv_bo_unmap(struct gdev_drv_bo *drv_bo)
{
	struct nouveau_bo *bo = (struct nouveau_bo *)drv_bo->priv;

	if (bo->kmap.bo) /* dirty validation.. */
		nouveau_bo_unmap(bo);
	else
		return -ENOENT;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_unmap);

int gdev_drv_read32(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint32_t *p)
{
	if (drv_bo->map)
		*p = ioread32_native(drv_bo->map + offset);
	else
		return -EINVAL;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_read32);

int gdev_drv_write32(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint32_t val)
{
	if (drv_bo->map)
		iowrite32_native(val, drv_bo->map + offset);
	else
		return -EINVAL;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_write32);

int gdev_drv_read(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t size, void *buf)
{
	if (drv_bo->map)
		memcpy_fromio(buf, drv_bo->map + offset, size);
	else
		return -EINVAL;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_read);

int gdev_drv_write(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t size, const void *buf)
{
	if (drv_bo->map)
		memcpy_toio(drv_bo->map + offset, buf, size);
	else
		return -EINVAL;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_write);

int gdev_drv_getdevice(int *count)
{
	*count = nouveau_device_count;
	return 0;
}
EXPORT_SYMBOL(gdev_drv_getdevice);

int gdev_drv_getdrm(int minor, struct drm_device **pptr)
{
	if (minor < nouveau_device_count) {
		if (nouveau_drm[minor]) {
			*pptr = nouveau_drm[minor];
			return 0;
		}
	}
	
	*pptr = NULL;

	return -ENODEV;
}
EXPORT_SYMBOL(gdev_drv_getdrm);

int gdev_drv_getparam(struct drm_device *drm, uint32_t type, uint64_t *res)
{
	struct drm_nouveau_getparam getparam;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	int ret = 0;

	switch (type) {
	case GDEV_DRV_GETPARAM_MP_COUNT:
		if ((dev_priv->chipset & 0xf0) == 0xc0) {
			struct nvc0_graph_priv *priv = nv_engine(drm, NVOBJ_ENGINE_GR);
			*res = priv->tp_total;
		}
		else {
			*res = 0;
			ret = -EINVAL;
		}
		break;
	case GDEV_DRV_GETPARAM_FB_SIZE:
		getparam.param = NOUVEAU_GETPARAM_FB_SIZE;
		ret = nouveau_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_AGP_SIZE:
		getparam.param = NOUVEAU_GETPARAM_AGP_SIZE;
		ret = nouveau_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_CHIPSET_ID:
		getparam.param = NOUVEAU_GETPARAM_CHIPSET_ID;
		ret = nouveau_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_BUS_TYPE:
		getparam.param = NOUVEAU_GETPARAM_BUS_TYPE;
		ret = nouveau_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_PCI_VENDOR:
		getparam.param = NOUVEAU_GETPARAM_PCI_VENDOR;
		ret = nouveau_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_PCI_DEVICE:
		getparam.param = NOUVEAU_GETPARAM_PCI_DEVICE;
		ret = nouveau_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	default:
		ret = -EINVAL;
	}

	return ret;
}
EXPORT_SYMBOL(gdev_drv_getparam);

int gdev_drv_getaddr(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t *addr)
{
	struct nouveau_bo *bo = (struct nouveau_bo *)drv_bo->priv;
	int page = offset / PAGE_SIZE;
	uint32_t x = offset - page * PAGE_SIZE;

	if (drv_bo->map) {
		if (bo->bo.mem.mem_type & TTM_PL_TT)
			*addr = ((struct ttm_dma_tt *)bo->bo.ttm)->dma_address[page] + x;
		else
			*addr = bo->bo.mem.bus.base + bo->bo.mem.bus.offset + x;
	}
	else {
		*addr = 0;
	}

	return 0;
}
EXPORT_SYMBOL(gdev_drv_getaddr);

int gdev_drv_setnotify(void (*func)(int subc, uint32_t data))
{
	nouveau_callback_notify = func;
	return 0;
}
EXPORT_SYMBOL(gdev_drv_setnotify);

int gdev_drv_unsetnotify(void (*func)(int subc, uint32_t data))
{
	if (nouveau_callback_notify != func)
		return -EINVAL;
	nouveau_callback_notify = NULL;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_unsetnotify);
