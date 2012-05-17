/*
 * CDDL HEADER START
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can obtain a copy of the license at usr/src/OPENSOLARIS.LICENSE
 * or http://www.opensolaris.org/os/licensing.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL HEADER in each
 * file and include the License file at usr/src/OPENSOLARIS.LICENSE.
 * If applicable, add the following below this CDDL HEADER, with the
 * fields enclosed by brackets "[]" replaced with your own identifying
 * information: Portions Copyright [yyyy] [name of copyright owner]
 *
 * CDDL HEADER END
 */

/*
 * Copyright 2010 PathScale Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include "drm.h"
#include "drmP.h"
#include "nouveau_drv.h"
#include "pscnv_engine.h"
#include "pscnv_chan.h"
#include "nv50_chan.h"
#include "nv50_vm.h"

struct nv84_crypt_engine {
	struct pscnv_engine base;
	spinlock_t lock;
};

struct nv84_crypt_chan {
	struct pscnv_bo *crctx;
};

#define nv84_crypt(x) container_of(x, struct nv84_crypt_engine, base)

static int nv84_crypt_oclasses[] = {
	0x74c1,
	0
};

void nv84_crypt_takedown(struct pscnv_engine *eng);
void nv84_crypt_irq_handler(struct drm_device *dev, int irq);
int nv84_crypt_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs);
int nv86_crypt_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs);
int nv84_crypt_chan_alloc(struct pscnv_engine *eng, struct pscnv_chan *ch);
void nv84_crypt_chan_free(struct pscnv_engine *eng, struct pscnv_chan *ch);
void nv84_crypt_chan_kill(struct pscnv_engine *eng, struct pscnv_chan *ch);
int nv84_crypt_chan_obj_new(struct pscnv_engine *eng, struct pscnv_chan *ch, uint32_t handle, uint32_t oclass, uint32_t flags);

int nv84_crypt_init(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nv84_crypt_engine *res = kzalloc(sizeof *res, GFP_KERNEL);

	if (!res) {
		NV_ERROR(dev, "PCRYPT: Couldn't allocate engine!\n");
		return -ENOMEM;
	}

	res->base.dev = dev;
	res->base.oclasses = nv84_crypt_oclasses;
	res->base.takedown = nv84_crypt_takedown;
	res->base.tlb_flush = nv84_crypt_tlb_flush;
	res->base.chan_alloc = nv84_crypt_chan_alloc;
	res->base.chan_kill = nv84_crypt_chan_kill;
	res->base.chan_free = nv84_crypt_chan_free;
	res->base.chan_obj_new = nv84_crypt_chan_obj_new;
	spin_lock_init(&res->lock);

	/* reset everything */
	nv_wr32(dev, 0x200, 0xffffbfff);
	nv_wr32(dev, 0x200, 0xffffffff);

	nv_wr32(dev, 0x102130, -1);	/* INTR */

#if 0
	nv_wr32(dev, 0x102080, 0x3150a);
	nv_wr32(dev, 0x10208c, 0x40f00f);
	nv_wr32(dev, 0x10208c, 0x40f007);
	nv_wr32(dev, 0x102300, 0x63);
	nv_wr32(dev, 0x102304, 0xf0000);
	nv_wr32(dev, 0x102308, 4);
	nv_wr32(dev, 0x10232c, 0x8010ffac);
	nv_wr32(dev, 0x102084, 0x10300);
#endif

	/* enable FIFO access */
	nv_wr32(dev, 0x10200c, 0x10);

#if 0
	/* mark no channel loaded */
	nv_wr32(dev, 0x102188, 0);
	nv_wr32(dev, 0x10218c, 0);
#endif

	dev_priv->engines[PSCNV_ENGINE_CRYPT] = &res->base;

	nouveau_irq_register(dev, 14, nv84_crypt_irq_handler);

	nv_wr32(dev, 0x102140, 0x18f);	/* INTR_EN */
	return 0;
}

void nv84_crypt_takedown(struct pscnv_engine *eng) {
	struct drm_nouveau_private *dev_priv = eng->dev->dev_private;
	nv_wr32(eng->dev, 0x102140, 0);	/* INTR_EN */
	nouveau_irq_unregister(eng->dev, 14);
	/* XXX */
	kfree(eng);
	dev_priv->engines[PSCNV_ENGINE_CRYPT] = 0;
}

int nv84_crypt_chan_alloc(struct pscnv_engine *eng, struct pscnv_chan *ch) {
	struct drm_device *dev = eng->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t hdr;
	uint64_t limit;
	int i;
	struct nv84_crypt_chan *crch = kzalloc(sizeof *crch, GFP_KERNEL);

	if (!crch) {
		NV_ERROR(dev, "PCRYPT: Couldn't allocate channel!\n");
		return -ENOMEM;
	}

	hdr = 0xa0;
	crch->crctx = pscnv_mem_alloc(dev, 0x100, PSCNV_GEM_CONTIG, 0, 0xc7c07e47);
	if (!crch->crctx) {
		NV_ERROR(dev, "PCRYPT: No VRAM for context!\n");
		kfree(crch);
		return -ENOMEM;
	}
	for (i = 0; i < 0x100; i += 4)
		nv_wv32(crch->crctx, i, 0);
	limit = crch->crctx->start + 0x100 - 1;

	nv_wv32(ch->bo, hdr + 0x00, 0x00190000);
	nv_wv32(ch->bo, hdr + 0x04, limit);
	nv_wv32(ch->bo, hdr + 0x08, crch->crctx->start);
	nv_wv32(ch->bo, hdr + 0x0c, (limit >> 32) << 24 | (crch->crctx->start >> 32));
	nv_wv32(ch->bo, hdr + 0x10, 0);
	nv_wv32(ch->bo, hdr + 0x14, 0);
	dev_priv->vm->bar_flush(dev);

	nv50_vs(ch->vspace)->engref[PSCNV_ENGINE_CRYPT]++;
	ch->engdata[PSCNV_ENGINE_CRYPT] = crch;
	return 0;
}

int nv84_crypt_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs) {
	return nv50_vm_flush(eng->dev, 0xa);
}

void nv84_crypt_chan_kill(struct pscnv_engine *eng, struct pscnv_chan *ch) {
	struct drm_device *dev = eng->dev;
	struct nv84_crypt_engine *crypt = nv84_crypt(eng);
	uint64_t start;
	unsigned long flags;
	spin_lock_irqsave(&crypt->lock, flags);
	start = nv04_timer_read(dev);
	/* disable PFIFO access */
	nv_wr32(dev, 0x10200c, 0);
	/* check if the channel we're freeing is active on PCRYPT. */
	if (nv_rd32(dev, 0x102188) == (0x80000000 | ch->bo->start >> 12)) {
		NV_INFO(dev, "Kicking channel %d off PCRYPT.\n", ch->cid);
		nv_wr32(dev, 0x102188, 0);
	}
	/* or maybe it was just going to be loaded in? */
	if (nv_rd32(dev, 0x10218c) == (0x80000000 | ch->bo->start >> 12)) {
		nv_wr32(dev, 0x10218c, 0);
	}
	/* back to normal state. */
	nv_wr32(dev, 0x10200c, 0x10);
	spin_unlock_irqrestore(&crypt->lock, flags);
}

void nv84_crypt_chan_free(struct pscnv_engine *eng, struct pscnv_chan *ch) {
	struct nv84_crypt_chan *crch = ch->engdata[PSCNV_ENGINE_CRYPT];
	pscnv_mem_free(crch->crctx);
	kfree(crch);
	nv50_vs(ch->vspace)->engref[PSCNV_ENGINE_CRYPT]--;
	ch->engdata[PSCNV_ENGINE_CRYPT] = 0;
	nv84_crypt_tlb_flush(eng, ch->vspace);
}

int nv84_crypt_chan_obj_new(struct pscnv_engine *eng, struct pscnv_chan *ch, uint32_t handle, uint32_t oclass, uint32_t flags) {
	uint32_t inst = nv50_chan_iobj_new(ch, 0x20);
	if (!inst) {
		return -ENOMEM;
	}
	nv_wv32(ch->bo, inst, oclass);
	nv_wv32(ch->bo, inst + 4, 0);
	nv_wv32(ch->bo, inst + 8, 0);
	nv_wv32(ch->bo, inst + 0xc, 0);
	nv_wv32(ch->bo, inst + 0x10, 0);
	nv_wv32(ch->bo, inst + 0x14, 0);
	nv_wv32(ch->bo, inst + 0x18, 0);
	nv_wv32(ch->bo, inst + 0x1c, 0);
	return pscnv_ramht_insert (&ch->ramht, handle, 0x500000 | inst >> 4);
}

void nv84_crypt_irq_handler(struct drm_device *dev, int irq) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nv84_crypt_engine *crypt = nv84_crypt(dev_priv->engines[PSCNV_ENGINE_CRYPT]);
	uint32_t status;
	unsigned long flags;
	uint32_t chandle, data, mthd;
	int cid;
	spin_lock_irqsave(&crypt->lock, flags);
	status = nv_rd32(dev, 0x102130);
	mthd = nv_rd32(dev, 0x102190);
	data = nv_rd32(dev, 0x102194);
	chandle = nv_rd32(dev, 0x102188) & 0x7fffffff;
	cid = pscnv_chan_handle_lookup(dev, chandle);
	if (cid == 128) {
		NV_ERROR(dev, "PCRYPT: UNKNOWN channel %x active!\n", chandle);
	}

	if (status & 1) {
		NV_ERROR(dev, "PCRYPT_INVALID_STATE: ch %d mthd %04x data %08x\n", cid, mthd, data);
		nv_wr32(dev, 0x102130, 1);
		status &= ~1;
	}

	if (status & 2) {
		NV_ERROR(dev, "PCRYPT_ILLEGAL_MTHD: ch %d mthd %04x data %08x\n", cid, mthd, data);
		nv_wr32(dev, 0x102130, 2);
		status &= ~2;
	}

	if (status & 4) {
		NV_ERROR(dev, "PCRYPT_ILLEGAL_CLASS: ch %d mthd %04x data %08x\n", cid, mthd, data);
		nv_wr32(dev, 0x102130, 4);
		status &= ~4;
	}

	if (status & 0x80) {
		NV_ERROR(dev, "PCRYPT_QUERY: ch %d mthd %04x data %08x\n", cid, mthd, data);
		nv_wr32(dev, 0x102130, 0x80);
		status &= ~0x80;
	}

	if (status & 0x100) {
		NV_ERROR(dev, "PCRYPT_FAULT: ch %d mthd %04x data %08x\n", cid, mthd, data);
		nv_wr32(dev, 0x102130, 0x100);
		status &= ~0x100;
	}

	if (status) {
		NV_ERROR(dev, "Unknown PCRYPT interrupt %08x\n", status);
		NV_ERROR(dev, "PCRYPT: ch %d mthd %04x data %08x\n", cid, mthd, data);
		nv_wr32(dev, 0x102130, status);
	}
	nv50_vm_trap(dev);
	nv_wr32(dev, 0x10200c, 0x10);
	spin_unlock_irqrestore(&crypt->lock, flags);
}
