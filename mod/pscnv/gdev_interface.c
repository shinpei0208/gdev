#include <linux/module.h>
#include "gdev_interface.h"
#include "nouveau_drv.h"
#include "pscnv_chan.h"
#include "pscnv_fifo.h"
#include "pscnv_gem.h"
#include "pscnv_ioctl.h"
#include "pscnv_mem.h"
#include "pscnv_vm.h"

#define VS_START 0x20000000
#define VS_END (1ull << 40)

extern int pscnv_device_count;
extern struct drm_device **pscnv_drm;
extern void (*pscnv_callback_notify)(int subc, uint32_t data);
extern uint32_t *nvc0_fifo_ctrl_ptr(struct drm_device *, struct pscnv_chan *);

int gdev_drv_vspace_alloc(struct drm_device *drm, uint64_t size, struct gdev_drv_vspace *drv_vspace)
{
	struct pscnv_vspace *vspace;

	if (!(vspace = pscnv_vspace_new(drm, size, 0, 0)))
		return -ENOMEM;

	vspace->filp = NULL; /* we don't need drm_filp in Gdev. */
	drv_vspace->priv = vspace;
	
	return 0;
}
EXPORT_SYMBOL(gdev_drv_vspace_alloc);

int gdev_drv_vspace_free(struct gdev_drv_vspace *drv_vspace)
{
	pscnv_vspace_unref(drv_vspace->priv);
	
	return 0;
}
EXPORT_SYMBOL(gdev_drv_vspace_free);

int gdev_drv_chan_alloc(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_chan *drv_chan)
{
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_chan *chan;
	struct pscnv_bo *ib_bo, *pb_bo;
	struct pscnv_mm_node *ib_mm, *pb_mm;
	uint32_t cid;
	volatile uint32_t *regs;
	uint32_t *ib_map, *pb_map;
	uint32_t ib_order, pb_order;
	uint64_t ib_base, pb_base;
	uint32_t ib_mask, pb_mask;
	uint32_t pb_size;
	int ret;
	
	if (!(chan = pscnv_chan_new(drm, vspace, 0))) {
		ret = -ENOMEM;
		goto fail_chan;
	}
	
	chan->filp = NULL; /* we don't need drm_filp in Gdev. */
	
	/* channel ID. */
	cid = chan->cid;

	/* FIFO indirect buffer setup. */
	ib_order = 9; /* it's hardcoded. */
	ib_bo = pscnv_mem_alloc(drm, 8 << ib_order, PSCNV_GEM_SYSRAM_SNOOP, 0, 0);
	if (!ib_bo) {
		ret = -ENOMEM;
		goto fail_ib;
	}
	ret = pscnv_vspace_map(vspace, ib_bo, VS_START, VS_END, 0, &ib_mm);
	if (ret)
		goto fail_ibmap;
	ib_map = vmap(ib_bo->pages, ib_bo->size >> PAGE_SHIFT, 0, PAGE_KERNEL);
	ib_base = ib_mm->start;
	ib_mask = (1 << ib_order) - 1;

	/* FIFO push buffer setup. */
	pb_order = 20; /* it's hardcoded. */
	pb_bo = pscnv_mem_alloc(drm, 1 << pb_order, PSCNV_GEM_SYSRAM_SNOOP, 0, 0);
	if (!pb_bo) {
		ret = -ENOMEM;
		goto fail_pb;
	}
	ret = pscnv_vspace_map(vspace, pb_bo, VS_START, VS_END, 0, &pb_mm);
	if (ret)
		goto fail_pbmap;
	pb_map = vmap(pb_bo->pages, pb_bo->size >> PAGE_SHIFT, 0, PAGE_KERNEL);
	pb_base = pb_mm->start;
	pb_mask = (1 << pb_order) - 1;
	pb_size = (1 << pb_order);

	/* FIFO init. */
	ret = dev_priv->fifo->chan_init_ib(chan, 0, 0, 1, ib_base, ib_order);
	if (ret)
		goto fail_fifo_init;

	switch (dev_priv->chipset & 0xf0) {
	case 0xc0:
		/* FIFO command queue registers. */
		regs = nvc0_fifo_ctrl_ptr(drm, chan);
		/* PCOPY engines. */
		ret = dev_priv->engines[PSCNV_ENGINE_COPY0]->chan_alloc(dev_priv->engines[PSCNV_ENGINE_COPY0], chan);
		if (ret)
			goto fail_pcopy0;
		ret = dev_priv->engines[PSCNV_ENGINE_COPY1]->chan_alloc(dev_priv->engines[PSCNV_ENGINE_COPY1], chan);
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

fail_fifo_reg:
fail_pcopy1:
	dev_priv->engines[PSCNV_ENGINE_COPY0]->chan_kill(dev_priv->engines[PSCNV_ENGINE_COPY0], chan);
fail_pcopy0:	
fail_fifo_init:
	vunmap(pb_map);
	pscnv_vspace_unmap(vspace, pb_mm->start);
fail_pbmap:
	pscnv_mem_free(pb_bo);
fail_pb:
	vunmap(ib_map);
	pscnv_vspace_unmap(vspace, ib_mm->start);
fail_ibmap:
	pscnv_mem_free(ib_bo);
fail_ib:
	pscnv_chan_unref(chan);
fail_chan:
	return ret;
}
EXPORT_SYMBOL(gdev_drv_chan_alloc);

int gdev_drv_chan_free(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_chan *drv_chan)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_chan *chan = (struct pscnv_chan *)drv_chan->priv;
	struct pscnv_bo *ib_bo = (struct pscnv_bo *)drv_chan->ib_bo;
	struct pscnv_bo *pb_bo = (struct pscnv_bo *)drv_chan->pb_bo;
	struct drm_nouveau_private *dev_priv = chan->dev->dev_private;
	uint32_t *ib_map = drv_chan->ib_map;
	uint32_t *pb_map = drv_chan->pb_map;
	uint64_t ib_base = drv_chan->ib_base;
	uint64_t pb_base = drv_chan->pb_base;

	dev_priv->engines[PSCNV_ENGINE_COPY0]->chan_kill(dev_priv->engines[PSCNV_ENGINE_COPY0], chan);
	dev_priv->engines[PSCNV_ENGINE_COPY1]->chan_kill(dev_priv->engines[PSCNV_ENGINE_COPY1], chan);

	vunmap(pb_map);
	pscnv_vspace_unmap(vspace, pb_base);
	pscnv_mem_free(pb_bo);
	vunmap(ib_map);
	pscnv_vspace_unmap(vspace, ib_base);
	pscnv_mem_free(ib_bo);

	pscnv_chan_unref(chan);

	return 0;
}
EXPORT_SYMBOL(gdev_drv_chan_free);

int gdev_drv_bo_alloc(struct drm_device *drm, uint64_t size, uint32_t drv_flags, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct pscnv_bo *bo;
	struct pscnv_mm_node *mm;
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	unsigned long bar1_start = pci_resource_start(drm->pdev, 1);
	uint32_t flags = 0;
	void *map;
	int ret;

	/* set memory type. */
	if (drv_flags & GDEV_DRV_BO_VRAM)
		flags |= PSCNV_GEM_VRAM_SMALL;
	if (drv_flags & GDEV_DRV_BO_SYSRAM)
		flags |= PSCNV_GEM_SYSRAM_SNOOP;
	if (drv_flags & GDEV_DRV_BO_MAPPABLE) {
		flags |= PSCNV_GEM_MAPPABLE;
		if (drv_flags & GDEV_DRV_BO_VRAM)
			flags |= PSCNV_GEM_CONTIG;
	}

	/* allocate physical memory space. */
	if (!(bo = pscnv_mem_alloc(drm, size, flags, 0, 0))) {
		ret = -ENOMEM;
		goto fail_bo;
	}

	/* allocate virtual address space, if requested. */
	if (drv_flags & GDEV_DRV_BO_VSPACE) {
		if (pscnv_vspace_map(vspace, bo, VS_START, VS_END, 0, &mm))
			goto fail_map;
		drv_bo->addr = mm->start;
	}
	else
		drv_bo->addr = 0;

	/* address, size, and map. */
	if (drv_flags & GDEV_DRV_BO_MAPPABLE) {
		if (drv_flags & GDEV_DRV_BO_SYSRAM) {
			if (bo->size > PAGE_SIZE)
				map = vmap(bo->pages, bo->size >> PAGE_SHIFT, 0, PAGE_KERNEL);
			else
				map = kmap(bo->pages[0]);
		}
		else {
			if (dev_priv->vm->map_user(bo))
				goto fail_map_user;
			if (!(map = ioremap(bar1_start + bo->map1->start, bo->size)))
				goto fail_ioremap;
		}
		drv_bo->map = map;
	}
	else {
		drv_bo->map = NULL;
	}
	drv_bo->size = bo->size;
	drv_bo->priv = bo;

	return 0;

fail_ioremap:
	pscnv_vspace_unmap_node(bo->map1);
fail_map_user:
	if (flags & GDEV_DRV_BO_VSPACE)
		pscnv_vspace_unmap(vspace, mm->start);
fail_map:
	pscnv_mem_free(bo);
fail_bo:
	return ret;

}
EXPORT_SYMBOL(gdev_drv_bo_alloc);

int gdev_drv_bo_free(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	uint64_t addr = drv_bo->addr;
	void *map = drv_bo->map;

	if (bo->flags & PSCNV_GEM_SYSRAM_SNOOP) {
		if (map) {
			if (bo->size > PAGE_SIZE)
				vunmap(map);
			else
				kunmap(map);
		}
		else
			return -ENOENT;
	}
	else if (bo->flags & PSCNV_GEM_CONTIG) {
		if (map)
			iounmap(map);
		else
			return -ENOENT;
	}

	if (addr)
		pscnv_vspace_unmap(vspace, addr);

	pscnv_mem_free(bo);

	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_free);

int gdev_drv_bo_bind(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct pscnv_mm_node *mm;
	void *map;

	if (pscnv_vspace_map(vspace, bo, VS_START, VS_END, 0,&mm))
		goto fail_map;

	if (bo->flags & PSCNV_GEM_SYSRAM_SNOOP) {
		if (bo->size > PAGE_SIZE)
			map = vmap(bo->pages, bo->size >> PAGE_SHIFT, 0, PAGE_KERNEL);
		else
			map = kmap(bo->pages[0]);
	}
	else
		map = NULL;

	drv_bo->addr = mm->start;
	drv_bo->size = bo->size;
	drv_bo->map = map;

	return 0;

fail_map:
	return -ENOMEM;
}
EXPORT_SYMBOL(gdev_drv_bo_bind);

int gdev_drv_bo_unbind(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;

	pscnv_vspace_unmap(vspace, drv_bo->addr);	
	
	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_unbind);

int gdev_drv_bo_map(struct drm_device *drm, struct gdev_drv_bo *drv_bo)
{
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	unsigned long bar1_start = pci_resource_start(drm->pdev, 1);
	void *map;

	if (dev_priv->vm->map_user(bo))
		goto fail_map_user;
	if (!(map = ioremap(bar1_start + bo->map1->start, bo->size)))
		goto fail_ioremap;

	bo->flags |= PSCNV_GEM_MAPPABLE;
	drv_bo->map = map;

	return 0;

fail_ioremap:
	pscnv_vspace_unmap_node(bo->map1);
fail_map_user:
	return -EIO;
}
EXPORT_SYMBOL(gdev_drv_bo_map);

int gdev_drv_bo_unmap(struct gdev_drv_bo *drv_bo)
{
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	void *map = drv_bo->map;

	iounmap(map);
	pscnv_vspace_unmap_node(bo->map1);
	bo->flags &= ~PSCNV_GEM_MAPPABLE;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_bo_unmap);

int gdev_drv_read32(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint32_t *p)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;

	if (drv_bo->map) {
		*p = ioread32_native(drv_bo->map + offset);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->read32(vspace, bo, drv_bo->addr + offset, p);
		mutex_unlock(&vspace->lock);
	}

	return 0;
}
EXPORT_SYMBOL(gdev_drv_read32);

int gdev_drv_write32(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint32_t val)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;

	if (drv_bo->map) {
		iowrite32_native(val, drv_bo->map + offset);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->write32(vspace, bo, drv_bo->addr + offset, val);
		mutex_unlock(&vspace->lock);
	}

	return 0;
}
EXPORT_SYMBOL(gdev_drv_write32);

int gdev_drv_read(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t size, void *buf)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;

	if (drv_bo->map) {
		memcpy_fromio(buf, drv_bo->map + offset, size);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->read(vspace, bo, drv_bo->addr + offset, buf, size);
		mutex_unlock(&vspace->lock);
	}

	return 0;
}
EXPORT_SYMBOL(gdev_drv_read);

int gdev_drv_write(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t size, const void *buf)
{
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;

	if (drv_bo->map) {
		memcpy_toio(drv_bo->map + offset, buf, size);
	}
	else {
		mutex_lock(&vspace->lock);
		dev_priv->vm->write(vspace, bo, drv_bo->addr + offset, buf, size);
		mutex_unlock(&vspace->lock);
	}

	return 0;
}
EXPORT_SYMBOL(gdev_drv_write);

int gdev_drv_getdevice(int *count)
{
	*count = pscnv_device_count;
	return 0;
}
EXPORT_SYMBOL(gdev_drv_getdevice);

int gdev_drv_getdrm(int minor, struct drm_device **pptr)
{
	if (minor < pscnv_device_count) {
		if (pscnv_drm[minor]) {
			*pptr = pscnv_drm[minor];
			return 0;
		}
	}
	
	*pptr = NULL;

	return -ENODEV;
}
EXPORT_SYMBOL(gdev_drv_getdrm);

int gdev_drv_getparam(struct drm_device *drm, uint32_t type, uint64_t *res)
{
	struct drm_pscnv_getparam getparam;
	int ret = 0;

	switch (type) {
	case GDEV_DRV_GETPARAM_MP_COUNT:
		getparam.param = PSCNV_GETPARAM_MP_COUNT;
		ret = pscnv_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_FB_SIZE:
		getparam.param = PSCNV_GETPARAM_FB_SIZE;
		ret = pscnv_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_AGP_SIZE:
		getparam.param = PSCNV_GETPARAM_AGP_SIZE;
		ret = pscnv_ioctl_getparam(drm, &getparam, NULL);
		*res = getparam.value;
		break;
	case GDEV_DRV_GETPARAM_CHIPSET_ID:
		getparam.param = PSCNV_GETPARAM_CHIPSET_ID;
		ret = pscnv_ioctl_getparam(drm, &getparam, NULL);
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
	struct pscnv_vspace *vspace = (struct pscnv_vspace *)drv_vspace->priv;
	struct pscnv_bo *bo = (struct pscnv_bo *)drv_bo->priv;
	struct drm_nouveau_private *dev_priv = drm->dev_private;
	int page = offset / PAGE_SIZE;
	uint32_t x = offset - page * PAGE_SIZE;

	if (bo->flags & PSCNV_GEM_MAPPABLE) {
		if (bo->flags & PSCNV_GEM_SYSRAM_SNOOP)
			*addr = bo->dmapages[page] + x;
		else
			*addr = pci_resource_start(drm->pdev, 1) + bo->map1->start + x;
	}
	else {
		*addr = dev_priv->vm->phys_getaddr(vspace, bo, drv_bo->addr + offset);
	}

	return 0;
}
EXPORT_SYMBOL(gdev_drv_getaddr);

int gdev_drv_setnotify(void (*func)(int subc, uint32_t data))
{
	pscnv_callback_notify = func;
	return 0;
}
EXPORT_SYMBOL(gdev_drv_setnotify);

int gdev_drv_unsetnotify(void (*func)(int subc, uint32_t data))
{
	if (pscnv_callback_notify != func)
		return -EINVAL;
	pscnv_callback_notify = NULL;

	return 0;
}
EXPORT_SYMBOL(gdev_drv_unsetnotify);
