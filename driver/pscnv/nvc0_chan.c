#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "nvc0_chan.h"
#include "pscnv_chan.h"
#include "nvc0_vm.h"

int nvc0_chan_new (struct pscnv_chan *ch)
{
	struct pscnv_vspace *vs = ch->vspace;
	struct drm_nouveau_private *dev_priv = ch->dev->dev_private;
	unsigned long flags;

	ch->bo = pscnv_mem_alloc(ch->dev, 0x1000, PSCNV_GEM_CONTIG,
			0, (ch->cid < 0 ? 0xc5a2ba7 : 0xc5a2f1f0));
	if (!ch->bo)
		return -ENOMEM;

	spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);
	ch->handle = ch->bo->start >> 12;
	spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);

	if (vs->vid != -3)
		dev_priv->vm->map_kernel(ch->bo);

	nv_wv32(ch->bo, 0x200, nvc0_vs(vs)->pd->start);
	nv_wv32(ch->bo, 0x204, nvc0_vs(vs)->pd->start >> 32);
	nv_wv32(ch->bo, 0x208, vs->size - 1);
	nv_wv32(ch->bo, 0x20c, (vs->size - 1) >> 32);

	if (ch->cid >= 0) {
		nv_wr32(ch->dev, 0x3000 + ch->cid * 8, (0x4 << 28) | ch->bo->start >> 12);
		spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);
		ch->handle = ch->bo->start >> 12;
		spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);
	}
	dev_priv->vm->bar_flush(ch->dev);
	return 0;
}

void nvc0_chan_free(struct pscnv_chan *ch)
{
	struct drm_nouveau_private *dev_priv = ch->dev->dev_private;
	unsigned long flags;
	spin_lock_irqsave(&dev_priv->chan->ch_lock, flags);
	ch->handle = 0;
	spin_unlock_irqrestore(&dev_priv->chan->ch_lock, flags);
	pscnv_mem_free(ch->bo);
}

void
nvc0_chan_takedown(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_chan_engine *che = nvc0_ch(dev_priv->chan);
	kfree(che);
}

int
nvc0_chan_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_chan_engine *che = kzalloc(sizeof *che, GFP_KERNEL);
	if (!che) {
		NV_ERROR(dev, "CH: Couldn't alloc engine\n");
		return -ENOMEM;
	}
	nv_wr32(dev, 0x200, nv_rd32(dev, 0x200) & 0xfffffeff);
	nv_wr32(dev, 0x200, nv_rd32(dev, 0x200) | 0x00000100);
	che->base.takedown = nvc0_chan_takedown;
	che->base.do_chan_new = nvc0_chan_new;
	che->base.do_chan_free = nvc0_chan_free;
	dev_priv->chan = &che->base;
	spin_lock_init(&dev_priv->chan->ch_lock);
	dev_priv->chan->ch_min = 1;
	dev_priv->chan->ch_max = 126;
	return 0;
}
