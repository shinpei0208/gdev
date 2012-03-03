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
#include "nouveau_grctx.h"
#include "pscnv_engine.h"
#include "pscnv_chan.h"
#include "nv50_chan.h"
#include "nv50_vm.h"

struct nv50_graph_engine {
	struct pscnv_engine base;
	uint32_t grctx_size;
};

struct nv50_graph_chan {
	struct pscnv_bo *grctx;
};

#define nv50_graph(x) container_of(x, struct nv50_graph_engine, base)

static int nv50_graph_oclasses[] = {
	/* NULL */
	0x0030, 
	/* m2mf */
	0x5039,
	/* NV01-style 2d */
	0x0012,
	0x0019,
	0x0043,
	0x0044,
	0x004a,
	0x0057,
	0x005d,
	0x005f,
	0x0072,
	0x305c,
	0x3064,
	0x3066,
	0x307b,
	0x308a,
	0x5062,
	0x5089,
	/* NV50-style 2d */
	0x502d,
	/* compute */
	0x50c0,
	/* 3d */
	0x5097,
	/* list terminator */
	0
};

static int nv84_graph_oclasses[] = {
	/* NULL */
	0x0030, 
	/* m2mf */
	0x5039,
	/* NV50-style 2d */
	0x502d,
	/* compute */
	0x50c0,
	/* 3d */
	0x5097,
	0x8297,
	/* list terminator */
	0
};

static int nva0_graph_oclasses[] = {
	/* NULL */
	0x0030, 
	/* m2mf */
	0x5039,
	/* NV50-style 2d */
	0x502d,
	/* compute */
	0x50c0,
	/* 3d */
	0x8397,
	/* list terminator */
	0
};

static int nva3_graph_oclasses[] = {
	/* NULL */
	0x0030, 
	/* m2mf */
	0x5039,
	/* NV50-style 2d */
	0x502d,
	/* compute */
	0x50c0,
	0x85c0,
	/* 3d */
	0x8597,
	/* list terminator */
	0
};

static int nvaf_graph_oclasses[] = {
	/* NULL */
	0x0030, 
	/* m2mf */
	0x5039,
	/* NV50-style 2d */
	0x502d,
	/* compute */
	0x50c0,
	0x85c0,
	/* 3d */
	0x8697,
	/* list terminator */
	0
};

void nv50_graph_takedown(struct pscnv_engine *eng);
void nv50_graph_irq_handler(struct drm_device *dev, int irq);
int nv50_graph_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs);
int nv86_graph_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs);
int nv50_graph_chan_alloc(struct pscnv_engine *eng, struct pscnv_chan *ch);
void nv50_graph_chan_free(struct pscnv_engine *eng, struct pscnv_chan *ch);
void nv50_graph_chan_kill(struct pscnv_engine *eng, struct pscnv_chan *ch);
int nv50_graph_chan_obj_new(struct pscnv_engine *eng, struct pscnv_chan *ch, uint32_t handle, uint32_t oclass, uint32_t flags);

int nv50_graph_init(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t units = nv_rd32(dev, 0x1540);
	struct nouveau_grctx ctx = {};
	int ret, i;
	uint32_t *cp;
	struct nv50_graph_engine *res = kzalloc(sizeof *res, GFP_KERNEL);

	if (!res) {
		NV_ERROR(dev, "PGRAPH: Couldn't allocate engine!\n");
		return -ENOMEM;
	}

	res->base.dev = dev;
	if (dev_priv->chipset == 0x50)
		res->base.oclasses = nv50_graph_oclasses;
	else if (dev_priv->chipset < 0xa0)
		res->base.oclasses = nv84_graph_oclasses;
	else if (dev_priv->chipset == 0xa0 ||
			(dev_priv->chipset >= 0xaa && dev_priv->chipset <= 0xac))
		res->base.oclasses = nva0_graph_oclasses;
	else if (dev_priv->chipset < 0xaa)
		res->base.oclasses = nva3_graph_oclasses;
	else
		res->base.oclasses = nvaf_graph_oclasses;
	res->base.takedown = nv50_graph_takedown;
	if (dev_priv->chipset == 0x86)
		res->base.tlb_flush = nv86_graph_tlb_flush;
	else
		res->base.tlb_flush = nv50_graph_tlb_flush;
	res->base.chan_alloc = nv50_graph_chan_alloc;
	res->base.chan_kill = nv50_graph_chan_kill;
	res->base.chan_free = nv50_graph_chan_free;
	res->base.chan_obj_new = nv50_graph_chan_obj_new;

	/* reset everything */
	nv_wr32(dev, 0x200, 0xffffefff);
	nv_wr32(dev, 0x200, 0xffffffff);

	/* reset and enable traps & interrupts */
	nv_wr32(dev, 0x400804, 0xc0000000);	/* DISPATCH */
	nv_wr32(dev, 0x406800, 0xc0000000);	/* M2MF */
	nv_wr32(dev, 0x400c04, 0xc0000000);	/* VFETCH */
	nv_wr32(dev, 0x401800, 0xc0000000);	/* STRMOUT */
	nv_wr32(dev, 0x405018, 0xc0000000);	/* CCACHE */
	nv_wr32(dev, 0x402000, 0xc0000000);	/* CLIPID */
	for (i = 0; i < 16; i++)
		if (units & 1 << i) {
			if (dev_priv->chipset < 0xa0) {
				nv_wr32(dev, 0x408900 + (i << 12), 0xc0000000);	/* TEX */
				nv_wr32(dev, 0x408e08 + (i << 12), 0xc0000000);	/* TPDMA */
				nv_wr32(dev, 0x408314 + (i << 12), 0xc0000000);	/* MPC */
			} else {
				nv_wr32(dev, 0x408600 + (i << 11), 0xc0000000);	/* TEX */
				nv_wr32(dev, 0x408708 + (i << 11), 0xc0000000);	/* TPDMA */
				nv_wr32(dev, 0x40831c + (i << 11), 0xc0000000);	/* MPC */
			}
		}
	nv_wr32(dev, 0x400108, -1);	/* TRAP */
	nv_wr32(dev, 0x400100, -1);	/* INTR */

	/* set ctxprog flags */
	nv_wr32(dev, 0x400824, 0x00004000);

	/* enable FIFO access */
	/* XXX: figure out what exactly is bit 16. All I know is that it's
	 * needed for QUERYs to work. */
	nv_wr32(dev, 0x400500, 0x00010001);

	/* init ZCULL... or something */
	nv_wr32(dev, 0x402ca8, 0x00000800);

	/* init DEBUG regs */
	/* XXX: look at the other two regs and values everyone uses. pick something. */
	nv_wr32(dev, 0x40008c, 0x00000004);

	/* init and upload ctxprog */
	cp = ctx.data = kmalloc (512 * 4, GFP_KERNEL);
	if (!ctx.data) {
		NV_ERROR (dev, "PGRAPH: Couldn't allocate ctxprog!\n");
		kfree(res);
		return -ENOMEM;
	}
	ctx.ctxprog_max = 512;
	ctx.dev = dev;
	ctx.mode = NOUVEAU_GRCTX_PROG;
	if ((ret = nv50_grctx_init(&ctx))) {
		kfree(ctx.data);
		kfree(res);
		return ret;
	}
	res->grctx_size = ctx.ctxvals_pos * 4;
	nv_wr32(dev, 0x400324, 0);
	for (i = 0; i < ctx.ctxprog_len; i++)
		nv_wr32(dev, 0x400328, cp[i]);
	kfree(ctx.data);
	
	/* mark no channel loaded */
	/* XXX: is that fully correct? */
	nv_wr32(dev, 0x40032c, 0);
	nv_wr32(dev, 0x400784, 0);
	nv_wr32(dev, 0x400320, 4);

	dev_priv->engines[PSCNV_ENGINE_GRAPH] = &res->base;

	nouveau_irq_register(dev, 12, nv50_graph_irq_handler);

	nv_wr32(dev, 0x400138, -1);	/* TRAP_EN */
	nv_wr32(dev, 0x40013c, -1);	/* INTR_EN */
	return 0;
}

void nv50_graph_takedown(struct pscnv_engine *eng) {
	struct drm_nouveau_private *dev_priv = eng->dev->dev_private;
	nv_wr32(eng->dev, 0x400138, 0);	/* TRAP_EN */
	nv_wr32(eng->dev, 0x40013c, 0);	/* INTR_EN */
	nouveau_irq_unregister(eng->dev, 12);
	/* XXX */
	kfree(eng);
	dev_priv->engines[PSCNV_ENGINE_GRAPH] = 0;
}

int nv50_graph_chan_alloc(struct pscnv_engine *eng, struct pscnv_chan *ch) {
	struct drm_device *dev = eng->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nv50_graph_engine *graph = nv50_graph(eng);
	struct nouveau_grctx ctx = {};
	uint32_t hdr;
	uint64_t limit;
	int i;
	struct nv50_graph_chan *grch = kzalloc(sizeof *grch, GFP_KERNEL);

	if (!grch) {
		NV_ERROR(dev, "PGRAPH: Couldn't allocate channel!\n");
		return -ENOMEM;
	}

	if (dev_priv->chipset == 0x50)
		hdr = 0x200;
	else
		hdr = 0x20;
	grch->grctx = pscnv_mem_alloc(dev, graph->grctx_size, PSCNV_GEM_CONTIG, 0, 0x97c07e47);
	if (!grch->grctx) {
		NV_ERROR(dev, "PGRAPH: No VRAM for context!\n");
		kfree(grch);
		return -ENOMEM;
	}
	for (i = 0; i < graph->grctx_size; i += 4)
		nv_wv32(grch->grctx, i, 0);
	ctx.dev = dev;
	ctx.mode = NOUVEAU_GRCTX_VALS;
	ctx.data = grch->grctx;
	nv50_grctx_init(&ctx);
	limit = grch->grctx->start + graph->grctx_size - 1;

	nv_wv32(ch->bo, hdr + 0x00, 0x00190000);
	nv_wv32(ch->bo, hdr + 0x04, limit);
	nv_wv32(ch->bo, hdr + 0x08, grch->grctx->start);
	nv_wv32(ch->bo, hdr + 0x0c, (limit >> 32) << 24 | (grch->grctx->start >> 32));
	nv_wv32(ch->bo, hdr + 0x10, 0);
	nv_wv32(ch->bo, hdr + 0x14, 0);
	dev_priv->vm->bar_flush(dev);

	nv50_vs(ch->vspace)->engref[PSCNV_ENGINE_GRAPH]++;
	ch->engdata[PSCNV_ENGINE_GRAPH] = grch;
	return 0;
}

int nv50_graph_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs) {
	return nv50_vm_flush(eng->dev, 0);
}

int nv86_graph_tlb_flush(struct pscnv_engine *eng, struct pscnv_vspace *vs) {
	/* NV86 TLB fuckup special workaround. */
	struct drm_device *dev = eng->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint64_t start;
	/* initial guess... */
	uint32_t mask380 = 0xffffffff;
	uint32_t mask384 = 0xffffffff;
	uint32_t mask388 = 0xffffffff;
	int ret;
	unsigned long flags;
	spin_lock_irqsave(&dev_priv->context_switch_lock, flags);
	nv_wr32(dev, 0x400500, 0);
	start = nv04_timer_read(dev);
	while ((nv_rd32(dev, 0x400380) & mask380) || (nv_rd32(dev, 0x400384) & mask384) || (nv_rd32(dev, 0x400388) & mask388)) {
		if (nv04_timer_read(dev) - start >= 2000000000) {
			/* if you see this message, mask* above probably need to be adjusted to not contain the bits you see failing */
			NV_ERROR(dev, "PGRAPH: idle wait for TLB flush fail: %08x %08x %08x [%08x]!\n", nv_rd32(dev, 0x400380), nv_rd32(dev, 0x400384), nv_rd32(dev, 0x400388), nv_rd32(dev, 0x400700));
			break;
		}
	}
	ret = nv50_vm_flush(dev, 0);
	nv_wr32(dev, 0x400500, 0x10001);
	spin_unlock_irqrestore(&dev_priv->context_switch_lock, flags);
	return ret;
}

void nv50_graph_chan_kill(struct pscnv_engine *eng, struct pscnv_chan *ch) {
	struct drm_device *dev = eng->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint64_t start;
	unsigned long flags;
	spin_lock_irqsave(&dev_priv->context_switch_lock, flags);
	start = nv04_timer_read(dev);
	/* disable PFIFO access */
	nv_wr32(dev, 0x400500, 0);
	/* tell ctxprog to hang in sync point, if it's executing */
	nv_wr32(dev, 0x400830, 1);
	/* make sure that ctxprog either isn't executing, or is waiting at the
	 * sync point. */
	while ((nv_rd32(dev, 0x400300) & 1) && !(nv_rd32(dev, 0x400824) & 0x80000000)) {
		if (nv04_timer_read(dev) - start >= 2000000000) {
			NV_ERROR(dev, "ctxprog wait fail!\n");
			break;
		}
	}
	/* check if the channel we're freeing is active on PGRAPH. */
	if (nv_rd32(dev, 0x40032c) == (0x80000000 | ch->bo->start >> 12)) {
		NV_INFO(dev, "Kicking channel %d off PGRAPH.\n", ch->cid);
		/* DIE */
		nv_wr32(dev, 0x400040, -1);
		nv_wr32(dev, 0x400040, 0);
		/* no active channel now. */
		nv_wr32(dev, 0x40032c, 0);
		/* if ctxprog was running, rewind it to the beginning. if it
		 * wasn't, this has no effect. */
		nv_wr32(dev, 0x400310, 0);
	}
	/* or maybe it was just going to be loaded in? */
	if (nv_rd32(dev, 0x400330) == (0x80000000 | ch->bo->start >> 12)) {
		nv_wr32(dev, 0x400330, 0);
		nv_wr32(dev, 0x400310, 0);
	}
	/* back to normal state. */
	nv_wr32(dev, 0x400830, 0);
	nv_wr32(dev, 0x400500, 0x10001);
	spin_unlock_irqrestore(&dev_priv->context_switch_lock, flags);
}

void nv50_graph_chan_free(struct pscnv_engine *eng, struct pscnv_chan *ch) {
	struct nv50_graph_chan *grch = ch->engdata[PSCNV_ENGINE_GRAPH];
	pscnv_mem_free(grch->grctx);
	kfree(grch);
	nv50_vs(ch->vspace)->engref[PSCNV_ENGINE_GRAPH]--;
	ch->engdata[PSCNV_ENGINE_GRAPH] = 0;
	nv50_graph_tlb_flush(eng, ch->vspace);
}

int nv50_graph_chan_obj_new(struct pscnv_engine *eng, struct pscnv_chan *ch, uint32_t handle, uint32_t oclass, uint32_t flags) {
	uint32_t inst = nv50_chan_iobj_new(ch, 0x10);
	if (!inst) {
		return -ENOMEM;
	}
	nv_wv32(ch->bo, inst, oclass);
	nv_wv32(ch->bo, inst + 4, 0);
	nv_wv32(ch->bo, inst + 8, 0);
	nv_wv32(ch->bo, inst + 0xc, 0);
	return pscnv_ramht_insert (&ch->ramht, handle, 0x100000 | inst >> 4);
}

struct pscnv_enumval {
	int value;
	char *name;
	void *data;
};

static struct pscnv_enumval dispatch_errors[] = {
	{ 3, "INVALID_OPERATION", 0 },
	{ 4, "INVALID_VALUE", 0 },
	{ 5, "INVALID_ENUM", 0 },

	{ 8, "INVALID_OBJECT", 0 },
	{ 9, "READ_ONLY_OBJECT", 0 },
	{ 0xa, "SUPERVISOR_OBJECT", 0 },
	{ 0xb, "INVALID_ADDRESS_ALIGNMENT", 0 },
	{ 0xc, "INVALID_BITFIELD", 0 },
	{ 0xd, "BEGIN_END_ACTIVE", 0 },
	{ 0xe, "SEMANTIC_COLOR_BACK_OVER_LIMIT", 0 },
	{ 0xf, "VIEWPORT_ID_NEEDS_GP", 0 },
	{ 0x10, "RT_DOUBLE_BIND", 0 },
	{ 0x11, "RT_TYPES_MISMATCH", 0 },
	{ 0x12, "RT_LINEAR_WITH_ZETA", 0 },

	{ 0x15, "FP_TOO_FEW_REGS", 0 },
	{ 0x16, "ZETA_FORMAT_CSAA_MISMATCH", 0 },
	{ 0x17, "RT_LINEAR_WITH_MSAA", 0 },
	{ 0x18, "FP_INTERPOLANT_START_OVER_LIMIT", 0 },
	{ 0x19, "SEMANTIC_LAYER_OVER_LIMIT", 0 },
	{ 0x1a, "RT_INVALID_ALIGNMENT", 0 },
	{ 0x1b, "SAMPLER_OVER_LIMIT", 0 },
	{ 0x1c, "TEXTURE_OVER_LIMIT", 0 },

	{ 0x1e, "GP_TOO_MANY_OUTPUTS", 0 },
	{ 0x1f, "RT_BPP128_WITH_MS8", 0 },

	{ 0x21, "Z_OUT_OF_BOUNDS", 0 },

	{ 0x23, "M2MF_OUT_OF_BOUNDS", 0 },

	{ 0x27, "CP_MORE_PARAMS_THAN_SHARED", 0 },
	{ 0x28, "CP_NO_REG_SPACE_STRIPED", 0 },
	{ 0x29, "CP_NO_REG_SPACE_PACKED", 0 },
	{ 0x2a, "CP_NOT_ENOUGH_WARPS", 0 },
	{ 0x2b, "CP_BLOCK_SIZE_MISMATCH", 0 },
	{ 0x2c, "CP_NOT_ENOUGH_LOCAL_WARPS", 0 },
	{ 0x2d, "CP_NOT_ENOUGH_STACK_WARPS", 0 },
	{ 0x2e, "CP_NO_BLOCKDIM_LATCH", 0 },

	{ 0x31, "ENG2D_FORMAT_MISMATCH", 0 },

	{ 0x3f, "PRIMITIVE_ID_NEEDS_GP", 0 },

	{ 0x44, "SEMANTIC_VIEWPORT_OVER_LIMIT", 0 },
	{ 0x45, "SEMANTIC_COLOR_FRONT_OVER_LIMIT", 0 },
	{ 0x46, "LAYER_ID_NEEDS_GP", 0 },
	{ 0x47, "SEMANTIC_CLIP_OVER_LIMIT", 0 },
	{ 0x48, "SEMANTIC_PTSZ_OVER_LIMIT", 0 },

	{ 0, 0, 0 },
};

static struct pscnv_enumval *pscnv_enum_find (struct pscnv_enumval *list, int val) {
	while (list->value != val && list->name)
		list++;
	if (list->name)
		return list;
	else
		return 0;
}

void nv50_graph_tex_trap(struct drm_device *dev, int cid, int tp) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t staddr, status;
	uint32_t e04, e08, e0c, e10;
	uint64_t addr;
	if (dev_priv->chipset < 0xa0)
		staddr = 0x408900 + tp * 0x1000;
	else
		staddr = 0x408600 + tp * 0x800;
	status = nv_rd32(dev, staddr) & 0x7fffffff;
	e04 = nv_rd32(dev, staddr + 4);
	e08 = nv_rd32(dev, staddr + 8);
	e0c = nv_rd32(dev, staddr + 0xc);
	e10 = nv_rd32(dev, staddr + 0x10);
	addr = (uint64_t)e08 << 8;
	if (!(status & 1)) { // seems always set...
		NV_ERROR(dev, "PGRAPH_TRAP_TEXTURE: ch %d TP %d status %08x [no 1!]\n", cid, tp, status);
	}
	status &= ~1;
	if (status & 2) {
		NV_ERROR(dev, "PGRAPH_TRAP_TEXTURE: ch %d TP %d FAULT at %llx\n", cid, tp, addr);
		status &= ~2;
	}
	if (status & 4) {
		NV_ERROR(dev, "PGRAPH_TRAP_TEXTURE: ch %d TP %d STORAGE_TYPE_MISMATCH type %02x\n", cid, tp, e10 >> 5 & 0x7f);
		status &= ~4;
	}
	if (status & 8) {
		NV_ERROR(dev, "PGRAPH_TRAP_TEXTURE: ch %d TP %d LINEAR_MISMATCH type %02x\n", cid, tp, e10 >> 5 & 0x7f);
		status &= ~8;
	}
	if (status & 0x20) {
		NV_ERROR(dev, "PGRAPH_TRAP_TEXTURE: ch %d TP %d WRONG_MEMTYPE type %02x\n", cid, tp, e10 >> 5 & 0x7f);
		status &= ~0x20;
	}
	if (status) {
		NV_ERROR(dev, "PGRAPH_TRAP_TEXTURE: ch %d TP %d status %08x\n", cid, tp, status);
	}
	NV_ERROR(dev, "magic: %08x %08x %08x %08x\n", e04, e08, e0c, e10);
	nv_wr32(dev, staddr, 0xc0000000);
}

void nv50_graph_mp_trap(struct drm_device *dev, int cid, int tp, int mp) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t mpaddr, mp10, status, pc, oplo, ophi;
	if (dev_priv->chipset < 0xa0)
		mpaddr = 0x408200 + tp * 0x1000 + mp * 0x80;
	else
		mpaddr = 0x408100 + tp * 0x800 + mp * 0x80;
	mp10 = nv_rd32(dev, mpaddr + 0x10);
	status = nv_rd32(dev, mpaddr + 0x14);
	nv_rd32(dev, mpaddr + 0x20);
	pc = nv_rd32(dev, mpaddr + 0x24);
	oplo = nv_rd32(dev, mpaddr + 0x70);
	ophi = nv_rd32(dev, mpaddr + 0x74);
	if (!status)
		return;
	if (status & 1) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d STACK_UNDERFLOW at %06x warp %d op %08x %08x\n", cid, tp, mp, pc & 0xffffff, pc >> 24, oplo, ophi);
		status &= ~1;
	}
	if (status & 2) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d STACK_MISMATCH at %06x warp %d op %08x %08x\n", cid, tp, mp, pc & 0xffffff, pc >> 24, oplo, ophi);
		status &= ~2;
	}
	if (status & 4) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d QUADON_ACTIVE at %06x warp %d op %08x %08x\n", cid, tp, mp, pc & 0xffffff, pc >> 24, oplo, ophi);
		status &= ~4;
	}
	if (status & 8) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d TIMEOUT at %06x warp %d op %08x %08x\n", cid, tp, mp, pc & 0xffffff, pc >> 24, oplo, ophi);
		status &= ~8;
	}
	if (status & 0x10) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d INVALID_OPCODE at %06x warp %d op %08x %08x\n", cid, tp, mp, pc & 0xffffff, pc >> 24, oplo, ophi);
		status &= ~0x10;
	}
	if (status & 0x40) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d BREAKPOINT at %06x warp %d op %08x %08x\n", cid, tp, mp, pc & 0xffffff, pc >> 24, oplo, ophi);
		status &= ~0x40;
	}
	if (status) {
		NV_ERROR(dev, "PGRAPH_TRAP_MP: ch %d TP %d MP %d status %08x at %06x warp %d op %08x %08x\n", cid, tp, mp, status, pc & 0xffffff, pc >> 24, oplo, ophi);
	}
	nv_wr32(dev, mpaddr + 0x10, mp10);
	nv_wr32(dev, mpaddr + 0x14, 0);
}

void nv50_graph_mpc_trap(struct drm_device *dev, int cid, int tp) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t staddr, status;
	if (dev_priv->chipset < 0xa0)
		staddr = 0x408314 + tp * 0x1000;
	else
		staddr = 0x40831c + tp * 0x800;
	status = nv_rd32(dev, staddr) & 0x7fffffff;
	if (status & 1) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d LOCAL_LIMIT_READ\n", cid, tp);
		status &= ~1;
	}
	if (status & 0x10) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d LOCAL_LIMIT_WRITE\n", cid, tp);
		status &= ~0x10;
	}
	if (status & 0x40) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d STACK_LIMIT\n", cid, tp);
		status &= ~0x40;
	}
	if (status & 0x100) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d GLOBAL_LIMIT_READ\n", cid, tp);
		status &= ~0x100;
	}
	if (status & 0x1000) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d GLOBAL_LIMIT_WRITE\n", cid, tp);
		status &= ~0x1000;
	}
	if (status & 0x10000) {
		nv50_graph_mp_trap(dev, cid, tp, 0);
		status &= ~0x10000;
	}
	if (status & 0x20000) {
		nv50_graph_mp_trap(dev, cid, tp, 1);
		status &= ~0x20000;
	}
	if (status & 0x40000) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d GLOBAL_LIMIT_RED\n", cid, tp);
		status &= ~0x40000;
	}
	if (status & 0x400000) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d GLOBAL_LIMIT_ATOM\n", cid, tp);
		status &= ~0x400000;
	}
	if (status & 0x4000000) {
		nv50_graph_mp_trap(dev, cid, tp, 2);
		status &= ~0x4000000;
	}
	if (status) {
		NV_ERROR(dev, "PGRAPH_TRAP_MPC: ch %d TP %d status %08x\n", cid, tp, status);
	}
	nv_wr32(dev, staddr, 0xc0000000);
}

void nv50_graph_tprop_trap(struct drm_device *dev, int cid, int tp) {
	static const char *const tprop_tnames[14] = {
		"RT0",
		"RT1",
		"RT2",
		"RT3",
		"RT4",
		"RT5",
		"RT6",
		"RT7",
		"ZETA",
		"LOCAL",
		"GLOBAL",
		"STACK",
		"DST2D",
		"???",
	};
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t staddr, status;
	uint32_t e0c, e10, e14, e18, e1c, e20, e24;
	int surf;
	uint64_t addr;
	if (dev_priv->chipset < 0xa0)
		staddr = 0x408e08 + tp * 0x1000;
	else
		staddr = 0x408708 + tp * 0x800;
	status = nv_rd32(dev, staddr) & 0x7fffffff;
	e0c = nv_rd32(dev, staddr + 4);
	e10 = nv_rd32(dev, staddr + 8);
	e14 = nv_rd32(dev, staddr + 0xc);
	e18 = nv_rd32(dev, staddr + 0x10);
	e1c = nv_rd32(dev, staddr + 0x14);
	e20 = nv_rd32(dev, staddr + 0x18);
	e24 = nv_rd32(dev, staddr + 0x1c);
	surf = e24 >> 0x18 & 0xf;
	addr = e10 | (uint64_t)e14 << 32;
	if (surf > 13)
		surf = 13;
	if (status & 0x4) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s SURF_WIDTH_OVERRUN\n", cid, tp, tprop_tnames[surf]);
		status &= ~0x4;
	}
	if (status & 0x8) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s SURF_HEIGHT_OVERRUN\n", cid, tp, tprop_tnames[surf]);
		status &= ~0x8;
	}
	if (status & 0x10) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s DST2D_FAULT at %llx\n", cid, tp, tprop_tnames[surf], addr);
		status &= ~0x10;
	}
	if (status & 0x20) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s ZETA_FAULT at %llx\n", cid, tp, tprop_tnames[surf], addr);
		status &= ~0x20;
	}
	if (status & 0x40) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s RT_FAULT at %llx\n", cid, tp, tprop_tnames[surf], addr);
		status &= ~0x40;
	}
	if (status & 0x80) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s CUDA_FAULT at %llx\n", cid, tp, tprop_tnames[surf], addr);
		status &= ~0x80;
	}
	if (status & 0x100) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s DST2D_STORAGE_TYPE_MISMATCH type %02x\n", cid, tp, tprop_tnames[surf], e24 & 0x7f);
		status &= ~0x100;
	}
	if (status & 0x200) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s ZETA_STORAGE_TYPE_MISMATCH type %02x\n", cid, tp, tprop_tnames[surf], e24 & 0x7f);
		status &= ~0x200;
	}
	if (status & 0x400) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s RT_STORAGE_TYPE_MISMATCH type %02x\n", cid, tp, tprop_tnames[surf], e24 & 0x7f);
		status &= ~0x400;
	}
	if (status & 0x800) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s DST2D_LINEAR_MISMATCH type %02x\n", cid, tp, tprop_tnames[surf], e24 & 0x7f);
		status &= ~0x800;
	}
	if (status & 0x1000) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s RT_LINEAR_MISMATCH type %02x\n", cid, tp, tprop_tnames[surf], e24 & 0x7f);
		status &= ~0x1000;
	}
	if (status) {
		NV_ERROR(dev, "PGRAPH_TRAP_TPROP: ch %d TP %d surf %s status %08x\n", cid, tp, tprop_tnames[surf], status);
	}
	NV_ERROR(dev, "magic: %08x %08x %08x %08x %08x %08x %08x\n",
			e0c, e10, e14, e18, e1c, e20, e24);
	nv_wr32(dev, staddr, 0xc0000000);
}

void nv50_graph_trap_handler(struct drm_device *dev, int cid) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t status = nv_rd32(dev, 0x400108);
	uint32_t ustatus;
	uint32_t units = nv_rd32(dev, 0x1540);
	int i;

	if (status & 0x001) {
		ustatus = nv_rd32(dev, 0x400804) & 0x7fffffff;
		if (ustatus & 0x00000001) {
			nv_wr32(dev, 0x400500, 0);
			if (nv_rd32(dev, 0x400808) & 0x80000000) {
				uint32_t class = nv_rd32(dev, 0x400814);
				uint32_t mthd = nv_rd32(dev, 0x400808) & 0x1ffc;
				uint32_t subc = (nv_rd32(dev, 0x400808) >> 16) & 0x7;
				uint32_t data = nv_rd32(dev, 0x40080c);
				NV_ERROR(dev, "PGRAPH_TRAP_DISPATCH: ch %d sub %d [%04x] mthd %04x data %08x\n", cid, subc, class, mthd, data);
				NV_INFO(dev, "PGRAPH_TRAP_DISPATCH: 400808: %08x\n", nv_rd32(dev, 0x400808));
				NV_INFO(dev, "PGRAPH_TRAP_DISPATCH: 400848: %08x\n", nv_rd32(dev, 0x400848));
				nv_wr32(dev, 0x400808, 0);
			} else {
				NV_ERROR(dev, "PGRAPH_TRAP_DISPATCH: No stuck command?\n");
			}
			nv_wr32(dev, 0x4008e8, nv_rd32(dev, 0x4008e8) & 3);
			nv_wr32(dev, 0x400848, 0);
		}
		if (ustatus & 0x00000002) {
			/* XXX: this one involves much more pain. */
			NV_ERROR(dev, "PGRAPH_TRAP_QUERY: ch %d.\n", cid);
		}
		if (ustatus & 0x00000004) {
			NV_ERROR(dev, "PGRAPH_TRAP_GRCTX_MMIO: ch %d. This is a kernel bug.\n", cid);
		}
		if (ustatus & 0x00000008) {
			NV_ERROR(dev, "PGRAPH_TRAP_GRCTX_XFER1: ch %d. This is a kernel bug.\n", cid);
		}
		if (ustatus & 0x00000010) {
			NV_ERROR(dev, "PGRAPH_TRAP_GRCTX_XFER2: ch %d. This is a kernel bug.\n", cid);
		}
		ustatus &= ~0x0000001f;
		if (ustatus)
			NV_ERROR(dev, "PGRAPH_TRAP_DISPATCH: Unknown ustatus 0x%08x on ch %d\n", ustatus, cid);
		nv_wr32(dev, 0x400804, 0xc0000000);
		nv_wr32(dev, 0x400108, 0x001);
		status &= ~0x001;
	}

	if (status & 0x002) {
		ustatus = nv_rd32(dev, 0x406800) & 0x7fffffff;
		if (ustatus & 1)
			NV_ERROR (dev, "PGRAPH_TRAP_M2MF_NOTIFY: ch %d %08x %08x %08x %08x\n",
				cid,
				nv_rd32(dev, 0x406804),
				nv_rd32(dev, 0x406808),
				nv_rd32(dev, 0x40680c),
				nv_rd32(dev, 0x406810));
		if (ustatus & 2)
			NV_ERROR (dev, "PGRAPH_TRAP_M2MF_IN: ch %d %08x %08x %08x %08x\n",
				cid,
				nv_rd32(dev, 0x406804),
				nv_rd32(dev, 0x406808),
				nv_rd32(dev, 0x40680c),
				nv_rd32(dev, 0x406810));
		if (ustatus & 4)
			NV_ERROR (dev, "PGRAPH_TRAP_M2MF_OUT: ch %d %08x %08x %08x %08x\n",
				cid,
				nv_rd32(dev, 0x406804),
				nv_rd32(dev, 0x406808),
				nv_rd32(dev, 0x40680c),
				nv_rd32(dev, 0x406810));
		ustatus &= ~0x00000007;
		if (ustatus)
			NV_ERROR(dev, "PGRAPH_TRAP_M2MF: Unknown ustatus 0x%08x on ch %d\n", ustatus, cid);
		/* No sane way found yet -- just reset the bugger. */
		nv_wr32(dev, 0x400040, 2);
		nv_wr32(dev, 0x400040, 0);
		nv_wr32(dev, 0x406800, 0xc0000000);
		nv_wr32(dev, 0x400108, 0x002);
		status &= ~0x002;
	}

	if (status & 0x004) {
		ustatus = nv_rd32(dev, 0x400c04) & 0x7fffffff;
		if (ustatus & 0x00000001) {
			NV_ERROR (dev, "PGRAPH_TRAP_VFETCH: ch %d\n", cid);
		}
		ustatus &= ~0x00000001;
		if (ustatus)
			NV_ERROR(dev, "PGRAPH_TRAP_VFETCH: Unknown ustatus 0x%08x on ch %d\n", ustatus, cid);
		nv_wr32(dev, 0x400c04, 0xc0000000);
		nv_wr32(dev, 0x400108, 0x004);
		status &= ~0x004;
	}

	if (status & 0x008) {
		ustatus = nv_rd32(dev, 0x401800) & 0x7fffffff;
		if (ustatus & 0x00000001) {
			NV_ERROR (dev, "PGRAPH_TRAP_STRMOUT: ch %d %08x %08x %08x %08x\n", cid,
				nv_rd32(dev, 0x401804),
				nv_rd32(dev, 0x401808),
				nv_rd32(dev, 0x40180c),
				nv_rd32(dev, 0x401810));
		}
		ustatus &= ~0x00000001;
		if (ustatus)
			NV_ERROR(dev, "PGRAPH_TRAP_STRMOUT: Unknown ustatus 0x%08x on ch %d\n", ustatus, cid);
		/* No sane way found yet -- just reset the bugger. */
		nv_wr32(dev, 0x400040, 0x80);
		nv_wr32(dev, 0x400040, 0);
		nv_wr32(dev, 0x401800, 0xc0000000);
		nv_wr32(dev, 0x400108, 0x008);
		status &= ~0x008;
	}

	if (status & 0x010) {
		ustatus = nv_rd32(dev, 0x405018) & 0x7fffffff;
		if (ustatus & 0x00000001) {
			NV_ERROR (dev, "PGRAPH_TRAP_CCACHE: ch %d\n", cid);
		}
		ustatus &= ~0x00000001;
		if (ustatus)
			NV_ERROR(dev, "PGRAPH_TRAP_CCACHE: Unknown ustatus 0x%08x on ch %d\n", ustatus, cid);
		nv_wr32(dev, 0x405018, 0xc0000000);
		nv_wr32(dev, 0x400108, 0x010);
		status &= ~0x010;
	}

	if (status & 0x020) {
		ustatus = nv_rd32(dev, 0x402000) & 0x7fffffff;
		if (ustatus & 0x00000001) {
			NV_ERROR (dev, "PGRAPH_TRAP_CLIPID: ch %d\n", cid);
		}
		ustatus &= ~0x00000001;
		if (ustatus)
			NV_ERROR(dev, "PGRAPH_TRAP_CLIPID: Unknown ustatus 0x%08x on ch %d\n", ustatus, cid);
		nv_wr32(dev, 0x402000, 0xc0000000);
		nv_wr32(dev, 0x400108, 0x020);
		status &= ~0x020;
	}

	if (status & 0x040) {
		for (i = 0; i < 16; i++)
			if (units & 1 << i) {
				if (dev_priv->chipset < 0xa0)
					ustatus = nv_rd32(dev, 0x408900 + i * 0x1000);
				else
					ustatus = nv_rd32(dev, 0x408600 + i * 0x800);
				if (ustatus & 0x7fffffff)
					nv50_graph_tex_trap(dev, cid, i);
			}
		nv_wr32(dev, 0x400108, 0x040);
		status &= ~0x040;
	}

	if (status & 0x080) {
		for (i = 0; i < 16; i++)
			if (units & 1 << i) {
				if (dev_priv->chipset < 0xa0)
					ustatus = nv_rd32(dev, 0x408314 + i * 0x1000);
				else
					ustatus = nv_rd32(dev, 0x40831c + i * 0x800);
				if (ustatus & 0x7fffffff)
					nv50_graph_mpc_trap(dev, cid, i);
			}
		nv_wr32(dev, 0x400108, 0x080);
		status &= ~0x080;
	}

	if (status & 0x100) {
		for (i = 0; i < 16; i++)
			if (units & 1 << i) {
				if (dev_priv->chipset < 0xa0)
					ustatus = nv_rd32(dev, 0x408e08 + i * 0x1000);
				else
					ustatus = nv_rd32(dev, 0x408708 + i * 0x800);
				if (ustatus & 0x7fffffff)
					nv50_graph_tprop_trap(dev, cid, i);
			}
		nv_wr32(dev, 0x400108, 0x100);
		status &= ~0x100;
	}

	/* XXX: per-TP traps. */

	if (status) {
		NV_ERROR(dev, "Unknown PGRAPH trap %08x on ch %d\n", status, cid);
		nv_wr32(dev, 0x400108, status);
	}
}

void nv50_graph_irq_handler(struct drm_device *dev, int irq) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t status;
	uint32_t st, chandle, addr, data, datah, ecode, class, subc, mthd;
	int cid;
	status = nv_rd32(dev, 0x400100);
	ecode = nv_rd32(dev, 0x400110);
	st = nv_rd32(dev, 0x400700);
	addr = nv_rd32(dev, 0x400704);
	mthd = addr & 0x1ffc;
	subc = (addr >> 16) & 7;
	data = nv_rd32(dev, 0x400708);
	datah = nv_rd32(dev, 0x40070c);
	chandle = nv_rd32(dev, 0x400784);
	class = nv_rd32(dev, 0x400814) & 0xffff;
	cid = pscnv_chan_handle_lookup(dev, chandle);
	if (cid == 128) {
		NV_ERROR(dev, "PGRAPH: UNKNOWN channel %x active!\n", chandle);
	}

	if (status & 0x00000001) {
		NV_ERROR(dev, "PGRAPH_NOTIFY: ch %d\n", cid);
		nv_wr32(dev, 0x400100, 0x00000001);
		status &= ~0x00000001;
	}
	if (status & 0x00000002) {
		NV_ERROR(dev, "PGRAPH_QUERY: ch %d\n", cid);
		nv_wr32(dev, 0x400100, 0x00000002);
		status &= ~0x00000002;
	}
	if (status & 0x00000004) {
		NV_ERROR(dev, "PGRAPH_SYNC: ch %d\n", cid);
		nv_wr32(dev, 0x400100, 0x00000004);
		status &= ~0x00000004;
	}
	if (status & 0x00000010) {
		NV_ERROR(dev, "PGRAPH_ILLEGAL_MTHD: ch %d sub %d [%04x] mthd %04x data %08x\n", cid, subc, class, mthd, data);
		nv_wr32(dev, 0x400100, 0x00000010);
		status &= ~0x00000010;
	}
	if (status & 0x00000020) {
		NV_ERROR(dev, "PGRAPH_ILLEGAL_CLASS: ch %d sub %d [%04x] mthd %04x data %08x\n", cid, subc, class, mthd, data);
		nv_wr32(dev, 0x400100, 0x00000020);
		status &= ~0x00000020;
	}
	if (status & 0x00000040) {
		NV_ERROR(dev, "PGRAPH_DOUBLE_NOTIFY: ch %d sub %d [%04x] mthd %04x data %08x\n", cid, subc, class, mthd, data);
		nv_wr32(dev, 0x400100, 0x00000040);
		status &= ~0x00000040;
	}
	if (status & 0x00010000) {
		NV_ERROR(dev, "PGRAPH_BUFFER_NOTIFY: ch %d\n", cid);
		nv_wr32(dev, 0x400100, 0x00010000);
		status &= ~0x00010000;
	}
	if (status & 0x00100000) {
		struct pscnv_enumval *ev;
		ev = pscnv_enum_find(dispatch_errors, ecode);
		if (ev)
			NV_ERROR(dev, "PGRAPH_DISPATCH_ERROR [%s]: ch %d sub %d [%04x] mthd %04x data %08x\n", ev->name, cid, subc, class, mthd, data);
		else {
			uint32_t base = (dev_priv->chipset > 0xa0 && dev_priv->chipset < 0xaa ? 0x404800 : 0x405400);
			int i;
			NV_ERROR(dev, "PGRAPH_DISPATCH_ERROR [%x]: ch %d sub %d [%04x] mthd %04x data %08x\n", ecode, cid, subc, class, mthd, data);
			for (i = 0; i < 0x400; i += 4)
				NV_ERROR(dev, "DD %06x: %08x\n", base + i, nv_rd32(dev, base + i));
		}
		nv_wr32(dev, 0x400100, 0x00100000);
		status &= ~0x00100000;
	}

	if (status & 0x00200000) {
		nv50_graph_trap_handler(dev, cid);
		nv_wr32(dev, 0x400100, 0x00200000);
		status &= ~0x00200000;
	}

	if (status & 0x01000000) {
		addr = nv_rd32(dev, 0x400808);
		subc = addr >> 16 & 7;
		mthd = addr & 0x1ffc;
		data = nv_rd32(dev, 0x40080c);
		NV_ERROR(dev, "PGRAPH_SINGLE_STEP: ch %d sub %d [%04x] mthd %04x data %08x\n", cid, subc, class, mthd, data);
		nv_wr32(dev, 0x400100, 0x01000000);
		status &= ~0x01000000;
	}

	if (status) {
		NV_ERROR(dev, "Unknown PGRAPH interrupt %08x\n", status);
		NV_ERROR(dev, "PGRAPH: ch %d sub %d [%04x] mthd %04x data %08x\n", cid, subc, class, mthd, data);
		nv_wr32(dev, 0x400100, status);
	}
	nv50_vm_trap(dev);
	nv_wr32(dev, 0x400500, 0x10001);
}
