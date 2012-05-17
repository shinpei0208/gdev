#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "nv50_chan.h"
#include "pscnv_chan.h"
#include "nv50_vm.h"

int nv50_chan_new (struct pscnv_chan *ch) {
	struct pscnv_vspace *vs = ch->vspace;
	struct drm_nouveau_private *dev_priv = ch->dev->dev_private;
	uint64_t size;
	uint32_t chan_pd;
	unsigned long flags;
	int i;
	/* determine size of underlying VO... for normal channels,
	 * allocate 64kiB since they have to store the objects
	 * heap. for the BAR fake channel, we'll only need two objects,
	 * so keep it minimal
	 */
	if (ch->cid >= 0)
		size = 0x10000;
	else if (dev_priv->chipset == 0x50)
		size = 0x6000;
	else
		size = 0x5000;
	ch->bo = pscnv_mem_alloc(vs->dev, size, PSCNV_GEM_CONTIG,
			0, (ch->cid == -1 ? 0xc5a2ba7 : 0xc5a2f1f0));
	if (!ch->bo)
		return -ENOMEM;

	spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);
	ch->handle = ch->bo->start >> 12;
	spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);

	if (vs->vid != -1)
		dev_priv->vm->map_kernel(ch->bo);

	mutex_lock(&vs->lock);
	list_add(&ch->vspace_list, &nv50_vs(vs)->chan_list);
	if (dev_priv->chipset == 0x50)
		chan_pd = NV50_CHAN_PD;
	else
		chan_pd = NV84_CHAN_PD;
	for (i = 0; i < NV50_VM_PDE_COUNT; i++) {
		if (nv50_vs(vs)->pt[i]) {
			nv_wv32(ch->bo, chan_pd + i * 8 + 4, nv50_vs(vs)->pt[i]->start >> 32);
			nv_wv32(ch->bo, chan_pd + i * 8, nv50_vs(vs)->pt[i]->start | 0x3);
		} else {
			nv_wv32(ch->bo, chan_pd + i * 8, 0);
		}
	}
	mutex_unlock(&vs->lock);

	ch->instpos = chan_pd + NV50_VM_PDE_COUNT * 8;

	if (ch->cid >= 0) {
		int i;
		ch->ramht.bo = ch->bo;
		ch->ramht.bits = 9;
		ch->ramht.offset = nv50_chan_iobj_new(ch, 8 << ch->ramht.bits);
		for (i = 0; i < (8 << ch->ramht.bits); i += 8)
			nv_wv32(ch->ramht.bo, ch->ramht.offset + i + 4, 0);

		if (dev_priv->chipset == 0x50) {
			ch->ramfc = 0;
		} else {
			/* actually, addresses of these two are NOT relative to
			 * channel struct on NV84+, and can be anywhere in VRAM,
			 * but we stuff them inside the channel struct anyway for
			 * simplicity. */
			ch->ramfc = nv50_chan_iobj_new(ch, 0x100);
			ch->cache = pscnv_mem_alloc(vs->dev, 0x1000, PSCNV_GEM_CONTIG,
					0, 0xf1f0cace);
			if (!ch->cache) {
				spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);
				ch->handle = 0;
				spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);
				pscnv_mem_free(ch->bo);
				return -ENOMEM;
			}
		}
		if (dev_priv->chipset != 0x50) {
			nv_wr32(vs->dev, 0x2600 + ch->cid * 4, (ch->bo->start + ch->ramfc) >> 8);
		} else {
			nv_wr32(vs->dev, 0x2600 + ch->cid * 4, ch->bo->start >> 12);
		}
	}
	dev_priv->vm->bar_flush(vs->dev);
	return 0;
}

int
nv50_chan_iobj_new(struct pscnv_chan *ch, uint32_t size) {
	/* XXX: maybe do this "properly" one day?
	 *
	 * Why we don't implement _del for instance objects:
	 *  - Usually, bounded const number of them is allocated
	 *    for any given channel, and the used set doesn't change
	 *    much during channel's lifetime
	 *  - Since instance objects are stored inside the main
	 *    VO of the channel, the storage will be freed on channel
	 *    close anyway
	 *  - We cannot easily tell what objects are currently in use
	 *    by PGRAPH and maybe other execution engines -- the user
	 *    could cheat us. Caching doesn't help either.
	 */
	int res;
	size += 0xf;
	size &= ~0xf;
	spin_lock(&ch->instlock);
	if (ch->instpos + size > ch->bo->size) {
		spin_unlock(&ch->instlock);
		return 0;
	}
	res = ch->instpos;
	ch->instpos += size;
	spin_unlock(&ch->instlock);
	return res;
}

/* XXX: we'll possibly want to break down type and/or add mysterious flags5
 * when we know more. */
int
nv50_chan_dmaobj_new(struct pscnv_chan *ch, uint32_t type, uint64_t start, uint64_t size) {
	struct drm_device *dev = ch->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint64_t end = start + size - 1;
	int res = nv50_chan_iobj_new (ch, 0x18);
	if (!res)
		return 0;

	nv_wv32(ch->bo, res + 0x00, type);
	nv_wv32(ch->bo, res + 0x04, end);
	nv_wv32(ch->bo, res + 0x08, start);
	nv_wv32(ch->bo, res + 0x0c, (end >> 32) << 24 | (start >> 32));
	nv_wv32(ch->bo, res + 0x10, 0);
	nv_wv32(ch->bo, res + 0x14, 0x80000);
	dev_priv->vm->bar_flush(dev);

	return res;
}

void nv50_chan_free(struct pscnv_chan *ch) {
	struct drm_nouveau_private *dev_priv = ch->dev->dev_private;
	unsigned long flags;
	spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);
	ch->handle = 0;
	spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);
	pscnv_mem_free(ch->bo);
	if (ch->cache)
		pscnv_mem_free(ch->cache);
	mutex_lock(&ch->vspace->lock);
	list_del(&ch->vspace_list);
	mutex_unlock(&ch->vspace->lock);
}

void
nv50_chan_takedown(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nv50_chan_engine *che = nv50_ch(dev_priv->chan);
	kfree(che);
}

int
nv50_chan_init(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nv50_chan_engine *che = kzalloc(sizeof *che, GFP_KERNEL);
	if (!che) {
		NV_ERROR(dev, "CH: Couldn't alloc engine\n");
		return -ENOMEM;
	}
	che->base.takedown = nv50_chan_takedown;
	che->base.do_chan_new = nv50_chan_new;
	che->base.do_chan_free = nv50_chan_free;
	dev_priv->chan = &che->base;
	spin_lock_init(&dev_priv->chan->ch_lock);
	dev_priv->chan->ch_min = 1;
	dev_priv->chan->ch_max = 126;
	return 0;
}
