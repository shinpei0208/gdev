/*
 * Copyright (C) Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: Ben Skeggs
 */

#include "drmP.h"
#include "nouveau_drv.h"
#include "pscnv_engine.h"
#include "pscnv_chan.h"
#include "nvc0_copy.h"
#include "nvc0_copy.fuc.h"
#include "nvc0_vm.h"

struct nvc0_copy_chan {
	struct pscnv_bo *bo;
	struct pscnv_mm_node *vm;
};

static int
nvc0_copy_chan_alloc(struct pscnv_engine *eng, struct pscnv_chan *ch)
{
	struct drm_device *dev = eng->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_copy_engine *pcopy = NVC0_COPY(eng);
	struct nvc0_copy_chan *coch = kzalloc(sizeof(*coch), GFP_KERNEL);
	uint32_t cookie = pcopy->fuc;
	int ret;

	coch->bo = pscnv_mem_alloc(dev, 256, PSCNV_GEM_CONTIG, 0, cookie);
	if (!coch->bo) {
		ret = -ENOMEM;
		goto fail_mem_alloc;
	}

	ret = dev_priv->vm->map_kernel(coch->bo);
	if (ret) {
		goto fail_map_kernel;
	}

	ret = pscnv_vspace_map(ch->vspace, coch->bo, 0x1000, (1ULL << 40) - 1, 0, 
						   &coch->vm);
	if (ret) {
		goto fail_vspace_map;
	}

	nv_wv32(ch->bo, pcopy->ctx + 0, coch->vm->start);
	nv_wv32(ch->bo, pcopy->ctx + 4, coch->vm->start >> 32);
	dev_priv->vm->bar_flush(dev);

	ch->engdata[PSCNV_ENGINE_COPY + pcopy->id] = coch;

	return 0;

fail_vspace_map:
fail_map_kernel:
	pscnv_mem_free(coch->bo);
fail_mem_alloc:
	kfree(coch);

	return ret;
}

void
nvc0_copy_chan_kill(struct pscnv_engine *eng, struct pscnv_chan *ch)
{
	/* FIXME */
}

static void
nvc0_copy_chan_free(struct pscnv_engine *eng, struct pscnv_chan *ch)
{
	struct drm_device *dev = eng->dev;
	struct nvc0_copy_engine *pcopy = NVC0_COPY(eng);
	struct nvc0_copy_chan *coch = ch->engdata[PSCNV_ENGINE_COPY + pcopy->id];
	uint32_t inst;

	inst  = (ch->bo->start >> 12);
	inst |= 0x40000000;

	/* disable fifo access */
	nv_wr32(dev, pcopy->fuc + 0x048, 0x00000000);
	/* mark channel as unloaded if it's currently active */
	if (nv_rd32(dev, pcopy->fuc + 0x050) == inst)
		nv_mask(dev, pcopy->fuc + 0x050, 0x40000000, 0x00000000);
	/* mark next channel as invalid if it's about to be loaded */
	if (nv_rd32(dev, pcopy->fuc + 0x054) == inst)
		nv_mask(dev, pcopy->fuc + 0x054, 0x40000000, 0x00000000);
	/* restore fifo access */
	nv_wr32(dev, pcopy->fuc + 0x048, 0x00000003);

	nv_wv32(ch->bo, pcopy->ctx + 0, 0x00000000);
	nv_wv32(ch->bo, pcopy->ctx + 4, 0x00000000);

	pscnv_vspace_unmap_node(coch->vm);
	pscnv_mem_free(coch->bo);
	kfree(coch);

	ch->engdata[PSCNV_ENGINE_COPY + pcopy->id] = NULL;
}

void
nvc0_copy_takedown(struct pscnv_engine *eng)
{
	struct drm_device *dev = eng->dev;
	struct nvc0_copy_engine *pcopy = NVC0_COPY(eng);

	nv_mask(dev, pcopy->fuc + 0x048, 0x00000003, 0x00000000);

	/* trigger fuc context unload */
	nv_wait(dev, pcopy->fuc + 0x008, 0x0000000c, 0x00000000);
	nv_mask(dev, pcopy->fuc + 0x054, 0x40000000, 0x00000000);
	nv_wr32(dev, pcopy->fuc + 0x000, 0x00000008);
	nv_wait(dev, pcopy->fuc + 0x008, 0x00000008, 0x00000000);

	nv_wr32(dev, pcopy->fuc + 0x014, 0xffffffff);

	nouveau_irq_unregister(dev, pcopy->irq);
	kfree(pcopy);
}

struct pscnv_copy_enum {
	int value;
	const char *name;
};

struct pscnv_copy_enum nvc0_copy_isr_error_name[] = {
	{ 0x0001, "ILLEGAL_MTHD" },
	{ 0x0002, "INVALID_ENUM" },
	{ 0x0003, "INVALID_BITFIELD" },
	{}
};

static void
nvc0_copy_isr(struct drm_device *dev, int engine)
{
	uint64_t inst;
	uint32_t disp, stat, chid, ssta, addr, mthd, subc, data;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_copy_engine *pcopy = NVC0_COPY(dev_priv->engines[engine]);
#define PCOPY_ERROR(name)												\
	NV_ERROR(dev, "%s: st %08x ch %d sub %d mthd %04x data %08x %08x/%08llx\n",\
			 name, stat, chid, subc, mthd, data, ssta, inst);

	disp = nv_rd32(dev, pcopy->fuc + 0x01c);
	stat = nv_rd32(dev, pcopy->fuc + 0x008) & disp & ~(disp >> 16);
	inst = (u64)(nv_rd32(dev, pcopy->fuc + 0x050) & 0x0fffffff) << 12;
	chid = -1;
	ssta = nv_rd32(dev, pcopy->fuc + 0x040) & 0x0000ffff;
	addr = nv_rd32(dev, pcopy->fuc + 0x040) >> 16;
	mthd = (addr & 0x07ff) << 2;
	subc = (addr & 0x3800) >> 11;
	data = nv_rd32(dev, pcopy->fuc + 0x044);

	if (stat & 0x00000040) {
		PCOPY_ERROR("PCOPY_DISPATCH");
		nv_wr32(dev, pcopy->fuc + 0x004, 0x00000040);
		stat &= ~0x00000040;
	}

	if (stat) {
		NV_INFO(dev, "PCOPY: unhandled intr 0x%08x\n", stat);
		nv_wr32(dev, pcopy->fuc + 0x004, stat);
	}
}

static void
nvc0_copy_isr_0(struct drm_device *dev, int irq)
{
	nvc0_copy_isr(dev, PSCNV_ENGINE_COPY0);
}

static void
nvc0_copy_isr_1(struct drm_device *dev, int irq)
{
	nvc0_copy_isr(dev, PSCNV_ENGINE_COPY1);
}

int
nvc0_copy_init(struct drm_device *dev, int engine)
{
	int i;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_copy_engine *pcopy = kzalloc(sizeof(*pcopy), GFP_KERNEL);

	if (!pcopy) {
		NV_ERROR(dev, "PCOPY%d: Couldn't allocate engine!\n", engine);
		return -ENOMEM;
	}
	NV_INFO(dev, "PCOPY%d: Initializing...\n", engine);

	dev_priv->engines[PSCNV_ENGINE_COPY + engine] = &pcopy->base;
	pcopy->base.dev = dev;
	pcopy->base.takedown = nvc0_copy_takedown;
	pcopy->base.chan_alloc = nvc0_copy_chan_alloc;
	pcopy->base.chan_kill = nvc0_copy_chan_kill;
	pcopy->base.chan_free = nvc0_copy_chan_free;
	spin_lock_init(&pcopy->lock);

	if (engine == 0) {
		pcopy->irq = 5;
		pcopy->pmc = 0x00000040;
		pcopy->fuc = 0x104000;
		pcopy->ctx = 0x0230;
		nouveau_irq_register(dev, pcopy->irq, nvc0_copy_isr_0);
	} else {
		pcopy->irq = 6;
		pcopy->pmc = 0x00000080;
		pcopy->fuc = 0x105000;
		pcopy->ctx = 0x0240;
		nouveau_irq_register(dev, pcopy->irq, nvc0_copy_isr_1);
	}
	pcopy->id = engine;

	nv_mask(dev, 0x000200, pcopy->pmc, 0x00000000);
	nv_mask(dev, 0x000200, pcopy->pmc, pcopy->pmc);
	nv_wr32(dev, pcopy->fuc + 0x014, 0xffffffff);

	nv_wr32(dev, pcopy->fuc + 0x1c0, 0x01000000);
	for (i = 0; i < sizeof(nvc0_pcopy_data) / 4; i++)
		nv_wr32(dev, pcopy->fuc + 0x1c4, nvc0_pcopy_data[i]);

	nv_wr32(dev, pcopy->fuc + 0x180, 0x01000000);
	for (i = 0; i < sizeof(nvc0_pcopy_code) / 4; i++) {
		if ((i & 0x3f) == 0)
			nv_wr32(dev, pcopy->fuc + 0x188, i >> 6);
		nv_wr32(dev, pcopy->fuc + 0x184, nvc0_pcopy_code[i]);
	}

	nv_wr32(dev, pcopy->fuc + 0x084, engine - PSCNV_ENGINE_COPY);
	nv_wr32(dev, pcopy->fuc + 0x10c, 0x00000000);
	nv_wr32(dev, pcopy->fuc + 0x104, 0x00000000); /* ENTRY */
	nv_wr32(dev, pcopy->fuc + 0x100, 0x00000002); /* TRIGGER */

	return 0;
}
