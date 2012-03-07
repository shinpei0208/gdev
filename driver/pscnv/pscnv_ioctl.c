#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "nouveau_reg.h"
#include "pscnv_ioctl.h"
#include "pscnv_vm.h"
#include "pscnv_chan.h"
#include "pscnv_fifo.h"
#include "pscnv_gem.h"
#include "nv50_chan.h"
#include "nvc0_graph.h"
#include "pscnv_kapi.h"

#ifdef PSCNV_KAPI_GETPARAM_BUS_TYPE
#define DEVICE_IS_AGP(dev) drm_device_is_agp(dev)
#define DEVICE_IS_PCIE(dev) drm_device_is_pcie(dev)
#else
#define DEVICE_IS_AGP(dev) drm_pci_device_is_agp(dev)
#define DEVICE_IS_PCIE(dev) pci_is_pcie(dev->pdev)
#endif

int pscnv_ioctl_getparam(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_pscnv_getparam *getparam = data;
	struct nvc0_graph_engine *nvc0_graph;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	switch (getparam->param) {
	case PSCNV_GETPARAM_MP_COUNT:
		if (dev_priv->chipset < 0xc0)
			goto fail;
		nvc0_graph = NVC0_GRAPH(dev_priv->engines[PSCNV_ENGINE_GRAPH]);
		getparam->value = nvc0_graph->tp_count; /* MPs == TPs */
		break;
	case PSCNV_GETPARAM_CHIPSET_ID:
		getparam->value = dev_priv->chipset;
		break;
	case PSCNV_GETPARAM_PCI_VENDOR:
		getparam->value = dev->pci_vendor;
		break;
	case PSCNV_GETPARAM_PCI_DEVICE:
		getparam->value = dev->pci_device;
		break;
	case PSCNV_GETPARAM_BUS_TYPE:
		if (DEVICE_IS_AGP(dev))
			getparam->value = NV_AGP;
		else if (DEVICE_IS_PCIE(dev))
			getparam->value = NV_PCIE;
		else
			getparam->value = NV_PCI;
		break;
	case PSCNV_GETPARAM_PTIMER_TIME:
		getparam->value = nv04_timer_read(dev);
		break;
	case PSCNV_GETPARAM_FB_SIZE:
		getparam->value = dev_priv->vram_size;
		break;
	case PSCNV_GETPARAM_GRAPH_UNITS:
		/* NV40 and NV50 versions are quite different, but register
		 * address is the same. User is supposed to know the card
		 * family anyway... */
		if (dev_priv->card_type >= NV_40 && dev_priv->card_type < NV_C0) {
			getparam->value = nv_rd32(dev, NV40_PMC_GRAPH_UNITS);
			break;
		}
		/* FALLTHRU */
	default:
		goto fail;
	}

	return 0;
fail:
	NV_ERROR(dev, "unknown parameter %lld\n", getparam->param);
	return -EINVAL;
}

int pscnv_ioctl_gem_new(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_gem_info *info = data;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	obj = pscnv_gem_new(dev, info->size, info->flags, info->tile_flags, info->cookie, info->user);
	if (!obj) {
		return -ENOMEM;
	}
	bo = obj->driver_private;

	/* could change due to page size align */
	info->size = bo->size;

	ret = drm_gem_handle_create(file_priv, obj, &info->handle);

	if (pscnv_gem_debug >= 1)
		NV_INFO(dev, "GEM handle %x is VO %x/%d\n", info->handle, bo->cookie, bo->serial);

	info->map_handle = (uint64_t)info->handle << 32;
	drm_gem_object_handle_unreference_unlocked (obj);
	return ret;
}

int pscnv_ioctl_gem_info(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_gem_info *info = data;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	int i;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	obj = drm_gem_object_lookup(dev, file_priv, info->handle);
	if (!obj)
		return -EBADF;

	bo = obj->driver_private;

	info->cookie = bo->cookie;
	info->flags = bo->flags;
	info->tile_flags = bo->tile_flags;
	info->size = obj->size;
	info->map_handle = (uint64_t)info->handle << 32;
	for (i = 0; i < ARRAY_SIZE(bo->user); i++)
		info->user[i] = bo->user[i];

	drm_gem_object_unreference_unlocked(obj);

	return 0;
}

struct pscnv_vspace *
pscnv_get_vspace(struct drm_device *dev, struct drm_file *file_priv, int vid)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	unsigned long flags;
	spin_lock_irqsave(&dev_priv->vm->vs_lock, flags);

	if (vid < 128 && vid >= 0 && dev_priv->vm->vspaces[vid] && dev_priv->vm->vspaces[vid]->filp == file_priv) {
		struct pscnv_vspace *res = dev_priv->vm->vspaces[vid];
		pscnv_vspace_ref(res);
		spin_unlock_irqrestore(&dev_priv->vm->vs_lock, flags);
		return res;
	}
	spin_unlock_irqrestore(&dev_priv->vm->vs_lock, flags);
	return 0;
}

int pscnv_ioctl_vspace_new(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_vspace_req *req = data;
	struct pscnv_vspace *vs;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_vspace_new(dev, 1ull << 40, 0, 0);
	if (!vs)
		return -ENOMEM;

	req->vid = vs->vid;

	vs->filp = file_priv;

	return 0;
}

int pscnv_ioctl_vspace_free(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_vspace_req *req = data;
	int vid = req->vid;
	struct pscnv_vspace *vs;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, vid);
	if (!vs)
		return -ENOENT;

	vs->filp = 0;
	pscnv_vspace_unref(vs);
	pscnv_vspace_unref(vs);

	return 0;
}

int pscnv_ioctl_vspace_map(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_vspace_map *req = data;
	struct pscnv_vspace *vs;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_mm_node *map;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	ret = pscnv_vspace_map(vs, bo, req->start, req->end, req->back, &map);
	if (!ret)
		req->offset = map->start;

	pscnv_vspace_unref(vs);

	return ret;
}

int pscnv_ioctl_vspace_unmap(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_vspace_unmap *req = data;
	struct pscnv_vspace *vs;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;


	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	ret = pscnv_vspace_unmap(vs, req->offset);

	pscnv_vspace_unref(vs);

	return ret;
}

void pscnv_vspace_cleanup(struct drm_device *dev, struct drm_file *file_priv) {
	int vid;
	struct pscnv_vspace *vs;

	for (vid = 0; vid < 128; vid++) {
		vs = pscnv_get_vspace(dev, file_priv, vid);
		if (!vs)
			continue;
		vs->filp = 0;
		pscnv_vspace_unref(vs);
		pscnv_vspace_unref(vs);
	}
}

struct pscnv_chan *
pscnv_get_chan(struct drm_device *dev, struct drm_file *file_priv, int cid)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	unsigned long flags;
	spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);

	if (cid < 128 && cid >= 0 && dev_priv->chan->chans[cid] && dev_priv->chan->chans[cid]->filp == file_priv) {
		struct pscnv_chan *res = dev_priv->chan->chans[cid];
		pscnv_chan_ref(res);
		spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);
		return res;
	}
	spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);
	return 0;
}

int pscnv_ioctl_chan_new(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_chan_new *req = data;
	struct pscnv_vspace *vs;
	struct pscnv_chan *ch;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	ch = pscnv_chan_new(dev, vs, 0);
	if (!ch) {
		pscnv_vspace_unref(vs);
		return -ENOMEM;
	}
	pscnv_vspace_unref(vs);

	req->cid = ch->cid;
	req->map_handle = 0xc0000000 | ch->cid << 16;

	ch->filp = file_priv;
	
	return 0;
}

int pscnv_ioctl_chan_free(struct drm_device *dev, void *data,
						struct drm_file *file_priv)
{
	struct drm_pscnv_chan_free *req = data;
	struct pscnv_chan *ch;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	ch = pscnv_get_chan(dev, file_priv, req->cid);
	if (!ch)
		return -ENOENT;

	ch->filp = 0;
	pscnv_chan_unref(ch);
	pscnv_chan_unref(ch);

	return 0;
}

int pscnv_ioctl_obj_vdma_new(struct drm_device *dev, void *data,
						struct drm_file *file_priv) {
	struct drm_pscnv_obj_vdma_new *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct pscnv_chan *ch;
	int ret;
	uint32_t oclass, inst;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	if (dev_priv->card_type != NV_50)
		return -ENOSYS;

	oclass = req->oclass;

	if (oclass != 2 && oclass != 3 && oclass != 0x3d)
		return -EINVAL;

	ch = pscnv_get_chan(dev, file_priv, req->cid);
	if (!ch)
		return -ENOENT;

	inst = nv50_chan_dmaobj_new(ch, 0x7fc00000 | oclass, req->start, req->size);
	if (!inst) {
		pscnv_chan_unref(ch);
		return -ENOMEM;
	}

	ret = pscnv_ramht_insert (&ch->ramht, req->handle, inst >> 4);

	pscnv_chan_unref(ch);

	return ret;
}

void pscnv_chan_cleanup(struct drm_device *dev, struct drm_file *file_priv) {
	int cid;
	struct pscnv_chan *ch;

	for (cid = 0; cid < 128; cid++) {
		ch = pscnv_get_chan(dev, file_priv, cid);
		if (!ch)
			continue;
		ch->filp = 0;
		pscnv_chan_unref(ch);
		pscnv_chan_unref(ch);
	}
}

int pscnv_ioctl_obj_eng_new(struct drm_device *dev, void *data,
						struct drm_file *file_priv) {
	struct drm_pscnv_obj_eng_new *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct pscnv_chan *ch;
	int ret;
	int i;
	uint32_t oclass = req->oclass;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	for (i = 0; i < PSCNV_ENGINES_NUM; i++)
		if (dev_priv->engines[i]) {
			uint32_t *pclass = dev_priv->engines[i]->oclasses;
			if (!pclass)
				continue;
			while (*pclass) {
				if (*pclass == oclass)
					goto found;
				pclass++;
			}
		}
	return -ENODEV;

found:
	ch = pscnv_get_chan(dev, file_priv, req->cid);
	if (!ch)
		return -ENOENT;

	if (!ch->engdata[i]) {
		ret = dev_priv->engines[i]->chan_alloc(dev_priv->engines[i], ch);
		if (ret) {
			pscnv_chan_unref(ch);
			return ret;
		}
	}

	ret = dev_priv->engines[i]->chan_obj_new(dev_priv->engines[i], ch, req->handle, oclass, req->flags);

	pscnv_chan_unref(ch);
	return ret;
}

int pscnv_ioctl_fifo_init(struct drm_device *dev, void *data,
						struct drm_file *file_priv) {
	struct drm_pscnv_fifo_init *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct pscnv_chan *ch;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	if (!dev_priv->fifo || !dev_priv->fifo->chan_init_dma)
		return -ENODEV;

	ch = pscnv_get_chan(dev, file_priv, req->cid);
	if (!ch)
		return -ENOENT;

	ret = dev_priv->fifo->chan_init_dma(ch, req->pb_handle, req->flags, req->slimask, req->pb_start);

	pscnv_chan_unref(ch);

	return ret;
}

int pscnv_ioctl_fifo_init_ib(struct drm_device *dev, void *data,
						struct drm_file *file_priv) {
	struct drm_pscnv_fifo_init_ib *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct pscnv_chan *ch;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	if (!dev_priv->fifo || !dev_priv->fifo->chan_init_ib)
		return -ENODEV;

	ch = pscnv_get_chan(dev, file_priv, req->cid);
	if (!ch)
		return -ENOENT;

	ret = dev_priv->fifo->chan_init_ib(ch, req->pb_handle, req->flags, req->slimask, req->ib_start, req->ib_order);

	pscnv_chan_unref(ch);

	return ret;
}

int pscnv_ioctl_vm_read32(struct drm_device *dev, void *data, struct drm_file *file_priv) 
{
	struct drm_pscnv_vm_rw32 *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;
	uint32_t val;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	mutex_lock(&vs->lock);
	ret = dev_priv->vm->read32(vs, bo, req->addr, &val);
	mutex_unlock(&vs->lock);
	req->val = val;

	pscnv_vspace_unref(vs);

	return ret;
}

int pscnv_ioctl_vm_write32(struct drm_device *dev, void *data, struct drm_file *file_priv)
{
	struct drm_pscnv_vm_rw32 *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	mutex_lock(&vs->lock);
	ret = dev_priv->vm->write32(vs, bo, req->addr, req->val);
	mutex_unlock(&vs->lock);

	pscnv_vspace_unref(vs);

	return ret;
}

int pscnv_ioctl_vm_read(struct drm_device *dev, void *data, struct drm_file *file_priv) 
{
	struct drm_pscnv_vm_rw *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;
	void *buf;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	/* req->size should be < 0x400000 */
	buf = kmalloc(req->size, GFP_KERNEL);
	if (!buf) {
		pscnv_vspace_unref(vs);
		return -ENOMEM;
	}

	mutex_lock(&vs->lock);
	ret = dev_priv->vm->read(vs, bo, req->addr, buf, req->size);
	mutex_unlock(&vs->lock);

	if (copy_to_user((void __user *) req->buf_rd, buf, req->size)) {
		pscnv_vspace_unref(vs);
		return -EINVAL;
	}

	kfree(buf);

	pscnv_vspace_unref(vs);

	return ret;
}

int pscnv_ioctl_vm_write(struct drm_device *dev, void *data, struct drm_file *file_priv)
{
	struct drm_pscnv_vm_rw *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;
	void *buf;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	/* req->size should be < 0x400000 */
	buf = kmalloc(req->size, GFP_KERNEL);
	if (!buf) {
		pscnv_vspace_unref(vs);
		return -ENOMEM;
	}

	if (copy_from_user(buf, (void __user *) req->buf_wr, req->size)) {
		pscnv_vspace_unref(vs);
		return -EINVAL;
	}

	mutex_lock(&vs->lock);
	ret = dev_priv->vm->write(vs, bo, req->addr, buf, req->size);
	mutex_unlock(&vs->lock);

	kfree(buf);

	pscnv_vspace_unref(vs);

	return ret;
}

int pscnv_ioctl_vm_map(struct drm_device *dev, void *data, struct drm_file *file_priv)
{
	struct drm_pscnv_vm_map *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;
	int ret;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	/* map the buffer object to BAR1. */
	ret = dev_priv->vm->map_user(bo);
	bo->flags |= PSCNV_GEM_MAPPABLE;

	pscnv_vspace_unref(vs);

	/* this handle will be used by mmap(). */
	req->map_handle = (uint64_t)req->handle << 32;

	return ret;
}

int pscnv_ioctl_vm_unmap(struct drm_device *dev, void *data, struct drm_file *file_priv)
{
	struct drm_pscnv_vm_map *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	if (dev_priv->vm_ok && bo->map1)
		pscnv_vspace_unmap_node(bo->map1);
	bo->flags &= ~PSCNV_GEM_MAPPABLE;

	pscnv_vspace_unref(vs);

	return 0;
}

int pscnv_ioctl_phys_getaddr(struct drm_device *dev, void *data, struct drm_file *file_priv)
{
	struct drm_pscnv_phys_getaddr *req = data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct drm_gem_object *obj;
	struct pscnv_bo *bo;
	struct pscnv_vspace *vs;
	int page;
	uint32_t x;

	NOUVEAU_CHECK_INITIALISED_WITH_RETURN;

	vs = pscnv_get_vspace(dev, file_priv, req->vid);
	if (!vs)
		return -ENOENT;

	obj = drm_gem_object_lookup(dev, file_priv, req->handle);
	if (!obj) {
		pscnv_vspace_unref(vs);
		return -EBADF;
	}

	bo = obj->driver_private;

	page = req->offset / PAGE_SIZE;
	x = req->offset - page * PAGE_SIZE;

	if (bo->flags & PSCNV_GEM_MAPPABLE) {
		if (bo->flags & PSCNV_GEM_SYSRAM_SNOOP)
			req->phys = bo->dmapages[page] + x;
		else
			req->phys = pci_resource_start(dev->pdev, 1) + bo->map1->start + x;
	}
	else {
		req->phys = dev_priv->vm->phys_getaddr(vs, bo, req->addr + req->offset);
	}

	pscnv_vspace_unref(vs);

	return 0;
}

#ifdef PSCNV_KAPI_DRM_IOCTL_DEF_DRV
struct drm_ioctl_desc nouveau_ioctls[] = {
	DRM_IOCTL_DEF_DRV(PSCNV_GETPARAM, pscnv_ioctl_getparam, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_GEM_NEW, pscnv_ioctl_gem_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_GEM_INFO, pscnv_ioctl_gem_info, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VSPACE_NEW, pscnv_ioctl_vspace_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VSPACE_FREE, pscnv_ioctl_vspace_free, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VSPACE_MAP, pscnv_ioctl_vspace_map, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VSPACE_UNMAP, pscnv_ioctl_vspace_unmap, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_CHAN_NEW, pscnv_ioctl_chan_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_CHAN_FREE, pscnv_ioctl_chan_free, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_OBJ_VDMA_NEW, pscnv_ioctl_obj_vdma_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_FIFO_INIT, pscnv_ioctl_fifo_init, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_OBJ_ENG_NEW, pscnv_ioctl_obj_eng_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_FIFO_INIT_IB, pscnv_ioctl_fifo_init_ib, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VM_READ32, pscnv_ioctl_vm_read32, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VM_WRITE32, pscnv_ioctl_vm_write32, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VM_READ, pscnv_ioctl_vm_read, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VM_WRITE, pscnv_ioctl_vm_write, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VM_MAP, pscnv_ioctl_vm_map, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_VM_UNMAP, pscnv_ioctl_vm_unmap, DRM_UNLOCKED),
	DRM_IOCTL_DEF_DRV(PSCNV_PHYS_GETADDR, pscnv_ioctl_phys_getaddr, DRM_UNLOCKED),
};
#elif defined(PSCNV_KAPI_DRM_IOCTL_DEF)
struct drm_ioctl_desc nouveau_ioctls[] = {
	DRM_IOCTL_DEF(DRM_PSCNV_GETPARAM, pscnv_ioctl_getparam, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_GEM_NEW, pscnv_ioctl_gem_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_GEM_INFO, pscnv_ioctl_gem_info, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VSPACE_NEW, pscnv_ioctl_vspace_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VSPACE_FREE, pscnv_ioctl_vspace_free, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VSPACE_MAP, pscnv_ioctl_vspace_map, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VSPACE_UNMAP, pscnv_ioctl_vspace_unmap, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_CHAN_NEW, pscnv_ioctl_chan_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_CHAN_FREE, pscnv_ioctl_chan_free, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_OBJ_VDMA_NEW, pscnv_ioctl_obj_vdma_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_FIFO_INIT, pscnv_ioctl_fifo_init, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_OBJ_ENG_NEW, pscnv_ioctl_obj_eng_new, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_FIFO_INIT_IB, pscnv_ioctl_fifo_init_ib, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VM_READ32, pscnv_ioctl_vm_read32, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VM_WRITE32, pscnv_ioctl_vm_write32, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VM_READ, pscnv_ioctl_vm_read, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VM_WRITE, pscnv_ioctl_vm_write, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VM_MAP, pscnv_ioctl_vm_map, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_VM_UNMAP, pscnv_ioctl_vm_unmap, DRM_UNLOCKED),
	DRM_IOCTL_DEF(DRM_PSCNV_PHYS_GETADDR, pscnv_ioctl_phys_getaddr, DRM_UNLOCKED),
};
#else
#error "Unknown IOCTLDEF method."
#endif

int nouveau_max_ioctl = DRM_ARRAY_SIZE(nouveau_ioctls);
