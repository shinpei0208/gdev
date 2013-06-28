/*
 * Copyright 2010 Christoph Bumiller.
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

#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "nouveau_reg.h"
#include "pscnv_mem.h"
#include "pscnv_vm.h"
#include "pscnv_chan.h"
#include "nvc0_vm.h"
#include <linux/list.h>

#define PSCNV_GEM_NOUSER 0x10 /* XXX */

int nvc0_vm_map_kernel(struct pscnv_bo *bo);
void nvc0_vm_takedown(struct drm_device *dev);
void nv84_vm_bar_flush(struct drm_device *dev);

int
nvc0_tlb_flush(struct pscnv_vspace *vs)
{
	struct drm_device *dev = vs->dev;
	uint32_t val;

	BUG_ON(!nvc0_vs(vs)->pd);

	NV_DEBUG(dev, "nvc0_tlb_flush 0x%010llx\n", nvc0_vs(vs)->pd->start);

	val = nv_rd32(dev, 0x100c80);

	nv_wr32(dev, 0x100cb8, nvc0_vs(vs)->pd->start >> 8);
	nv_wr32(dev, 0x100cbc, 0x80000000 | ((vs->vid == -3) ? 0x5 : 0x1));

	if (!nv_wait(dev, 0x100c80, ~0, val)) {
		NV_ERROR(vs->dev, "tlb flush timed out\n");
		return -EBUSY;
	}
	return 0;
}

static int
nvc0_vspace_fill_pde(struct pscnv_vspace *vs, struct nvc0_pgt *pgt)
{
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	const uint32_t size = NVC0_VM_SPTE_COUNT << (3 - pgt->limit);
	int i;
	uint32_t pde[2];

	pgt->bo[1] = pscnv_mem_alloc(vs->dev, size, PSCNV_GEM_CONTIG, 0, 0x59);
	if (!pgt->bo[1])
		return -ENOMEM;

	for (i = 0; i < size; i += 4)
		nv_wv32(pgt->bo[1], i, 0);

	pde[0] = pgt->limit << 2;
	pde[1] = (pgt->bo[1]->start >> 8) | 1;

	if (vs->vid != -3) {
		pgt->bo[0] = pscnv_mem_alloc(vs->dev, NVC0_VM_LPTE_COUNT * 8,
					      PSCNV_GEM_CONTIG, 0, 0x79);
		if (!pgt->bo[0])
			return -ENOMEM;

		nvc0_vm_map_kernel(pgt->bo[0]);
		nvc0_vm_map_kernel(pgt->bo[1]);

		for (i = 0; i < NVC0_VM_LPTE_COUNT * 8; i += 4)
			nv_wv32(pgt->bo[0], i, 0);

		pde[0] |= (pgt->bo[0]->start >> 8) | 1;
	}
	dev_priv->vm->bar_flush(vs->dev);

	nv_wv32(nvc0_vs(vs)->pd, pgt->pde * 8 + 0, pde[0]);
	nv_wv32(nvc0_vs(vs)->pd, pgt->pde * 8 + 4, pde[1]);

	dev_priv->vm->bar_flush(vs->dev);
	return nvc0_tlb_flush(vs);
}

static struct nvc0_pgt *
nvc0_vspace_pgt(struct pscnv_vspace *vs, unsigned int pde)
{
	struct nvc0_pgt *pt;
	struct list_head *pts = &nvc0_vs(vs)->ptht[NVC0_PDE_HASH(pde)];

	BUG_ON(pde >= NVC0_VM_PDE_COUNT);

	list_for_each_entry(pt, pts, head)
		if (pt->pde == pde)
			return pt;

	NV_DEBUG(vs->dev, "creating new page table: %i[%u]\n", vs->vid, pde);

	pt = kzalloc(sizeof *pt, GFP_KERNEL);
	if (!pt)
		return NULL;
	pt->pde = pde;
	pt->limit = 0;

	if (nvc0_vspace_fill_pde(vs, pt)) {
		kfree(pt);
		return NULL;
	}

	list_add_tail(&pt->head, pts);
	return pt;
}

void
nvc0_pgt_del(struct pscnv_vspace *vs, struct nvc0_pgt *pgt)
{
	pscnv_vram_free(pgt->bo[1]);
	if (pgt->bo[0])
		pscnv_vram_free(pgt->bo[0]);
	list_del(&pgt->head);

	nv_wv32(nvc0_vs(vs)->pd, pgt->pde * 8 + 0, 0);
	nv_wv32(nvc0_vs(vs)->pd, pgt->pde * 8 + 4, 0);

	kfree(pgt);
}

int
nvc0_vspace_do_unmap(struct pscnv_vspace *vs, uint64_t offset, uint64_t size)
{
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	uint32_t space;

	for (; size; offset += space) {
		struct nvc0_pgt *pt;
		int i, pte;

		pt = nvc0_vspace_pgt(vs, NVC0_PDE(offset));
		space = NVC0_VM_BLOCK_SIZE - (offset & NVC0_VM_BLOCK_MASK);
		if (space > size)
			space = size;
		size -= space;

		pte = NVC0_SPTE(offset);
		for (i = 0; i < (space >> NVC0_SPAGE_SHIFT) * 8; i += 4)
			nv_wv32(pt->bo[1], pte * 8 + i, 0);

		if (!pt->bo[0])
			continue;

		pte = NVC0_LPTE(offset);
		for (i = 0; i < (space >> NVC0_LPAGE_SHIFT) * 8; i += 4)
			nv_wv32(pt->bo[0], pte * 8 + i, 0);
	}
	dev_priv->vm->bar_flush(vs->dev);
	return nvc0_tlb_flush(vs);
}

static inline void
write_pt(struct pscnv_bo *pt, int pte, int count, uint64_t phys,
	 int psz, uint32_t pfl0, uint32_t pfl1)
{
	int i;
	uint32_t a = (phys >> 8) | pfl0;
	uint32_t b = pfl1;

	psz >>= 8;

	for (i = pte * 8; i < (pte + count) * 8; i += 8, a += psz) {
		nv_wv32(pt, i + 4, b);
		nv_wv32(pt, i + 0, a);
	}
}

int
nvc0_vspace_place_map (struct pscnv_vspace *vs, struct pscnv_bo *bo,
		       uint64_t start, uint64_t end, int back,
		       struct pscnv_mm_node **res)
{
	int flags = 0;

	if ((bo->flags & PSCNV_GEM_MEMTYPE_MASK) == PSCNV_GEM_VRAM_LARGE)
		flags = PSCNV_MM_LP;
	if (back)
		flags |= PSCNV_MM_FROMBACK;

	return pscnv_mm_alloc(vs->mm, bo->size, flags, start, end, res);
}

int
nvc0_vspace_do_map(struct pscnv_vspace *vs,
		   struct pscnv_bo *bo, uint64_t offset)
{
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	uint32_t pfl0, pfl1;
	struct pscnv_mm_node *reg;
	int i;

	pfl0 = 1;
	if (vs->vid >= 0 && (bo->flags & PSCNV_GEM_NOUSER))
		pfl0 |= 2;

	pfl1 = bo->tile_flags << 4;

	switch (bo->flags & PSCNV_GEM_MEMTYPE_MASK) {
	case PSCNV_GEM_SYSRAM_NOSNOOP:
		pfl1 |= 0x2;
		/* fall through */
	case PSCNV_GEM_SYSRAM_SNOOP:
	{
		unsigned int pde = NVC0_PDE(offset);
		unsigned int pte = (offset & NVC0_VM_BLOCK_MASK) >> PAGE_SHIFT;
		struct nvc0_pgt *pt = nvc0_vspace_pgt(vs, pde);
		pfl1 |= 0x5;
		for (i = 0; i < (bo->size >> PAGE_SHIFT); ++i) {
			uint64_t phys = bo->dmapages[i];
			nv_wv32(pt->bo[1], pte * 8 + 4, pfl1);
			nv_wv32(pt->bo[1], pte * 8 + 0, (phys >> 8) | pfl0);
			pte++;
			if ((pte & (NVC0_VM_BLOCK_MASK >> PAGE_SHIFT)) == 0) {
				pte = 0;
				pt = nvc0_vspace_pgt(vs, ++pde);
			}
		}
	}
		break;
	case PSCNV_GEM_VRAM_SMALL:
	case PSCNV_GEM_VRAM_LARGE:
		for (reg = bo->mmnode; reg; reg = reg->next) {
			uint32_t psh, psz;
			uint64_t phys = reg->start, size = reg->size;

			int s = (bo->flags & PSCNV_GEM_MEMTYPE_MASK) != PSCNV_GEM_VRAM_LARGE;
			if (vs->vid == -3)
				s = 1;
			psh = s ? NVC0_SPAGE_SHIFT : NVC0_LPAGE_SHIFT;
			psz = 1 << psh;

			while (size) {
				struct nvc0_pgt *pt;
				int pte, count;
				uint32_t space;

				space = NVC0_VM_BLOCK_SIZE -
					(offset & NVC0_VM_BLOCK_MASK);
				if (space > size)
					space = size;
				size -= space;

				pte = (offset & NVC0_VM_BLOCK_MASK) >> psh;
				count = space >> psh;
				pt = nvc0_vspace_pgt(vs, NVC0_PDE(offset));

				write_pt(pt->bo[s], pte, count, phys, psz, pfl0, pfl1);

				offset += space;
				phys += space;
			}
		}
		break;
	default:
		return -ENOSYS;
	}
	dev_priv->vm->bar_flush(vs->dev);
	return nvc0_tlb_flush(vs);
}

int nvc0_vspace_new(struct pscnv_vspace *vs) {
	int i, ret;

	if (vs->size > 1ull << 40)
		return -EINVAL;

	vs->engdata = kzalloc(sizeof(struct nvc0_vspace), GFP_KERNEL);
	if (!vs->engdata) {
		NV_ERROR(vs->dev, "VM: Couldn't alloc vspace eng\n");
		return -ENOMEM;
	}

	nvc0_vs(vs)->pd = pscnv_mem_alloc(vs->dev, NVC0_VM_PDE_COUNT * 8,
			PSCNV_GEM_CONTIG, 0, 0xdeadcafe);
	if (!nvc0_vs(vs)->pd) {
		kfree(vs->engdata);
		return -ENOMEM;
	}

	if (vs->vid != -3)
		nvc0_vm_map_kernel(nvc0_vs(vs)->pd);

	for (i = 0; i < NVC0_VM_PDE_COUNT; i++) {
		nv_wv32(nvc0_vs(vs)->pd, i * 8, 0);
		nv_wv32(nvc0_vs(vs)->pd, i * 8 + 4, 0);
	}
	
	for (i = 0; i < NVC0_PDE_HT_SIZE; ++i)
		INIT_LIST_HEAD(&nvc0_vs(vs)->ptht[i]);

	ret = pscnv_mm_init(vs->dev, 0, vs->size, 0x1000, 0x20000, 1, &vs->mm);
	if (ret) {
		pscnv_mem_free(nvc0_vs(vs)->pd);
		kfree(vs->engdata);
	}
	return ret;
}

void nvc0_vspace_free(struct pscnv_vspace *vs) {
	int i;
	for (i = 0; i < NVC0_PDE_HT_SIZE; i++) {
		struct nvc0_pgt *pgt, *save;
		list_for_each_entry_safe(pgt, save, &nvc0_vs(vs)->ptht[i], head)
			nvc0_pgt_del(vs, pgt);
	}
	pscnv_mem_free(nvc0_vs(vs)->pd);

	kfree(vs->engdata);
}

int nvc0_vm_map_user(struct pscnv_bo *bo) {
	struct drm_nouveau_private *dev_priv = bo->dev->dev_private;
	struct nvc0_vm_engine *vme = nvc0_vm(dev_priv->vm);
	if (bo->map1)
		return 0;
	return pscnv_vspace_map(vme->bar1vm, bo, 0, dev_priv->fb_size, 0, &bo->map1);
}

int nvc0_vm_map_kernel(struct pscnv_bo *bo) {
	struct drm_nouveau_private *dev_priv = bo->dev->dev_private;
	struct nvc0_vm_engine *vme = nvc0_vm(dev_priv->vm);
	if (bo->map3)
		return 0;
	return pscnv_vspace_map(vme->bar3vm, bo, 0, dev_priv->ramin_size, 0, &bo->map3);
}

uint64_t nvc0_vm_phys_getaddr(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr)
{
	int s = (bo->flags & PSCNV_GEM_MEMTYPE_MASK) != PSCNV_GEM_VRAM_LARGE;
	uint32_t psh = s ? NVC0_SPAGE_SHIFT : NVC0_LPAGE_SHIFT;
	unsigned int pde = NVC0_PDE(addr);
	unsigned int pte = (addr & NVC0_VM_BLOCK_MASK) >> psh;
	uint64_t vm_block = pte << psh;
	uint64_t offset = (addr & NVC0_VM_BLOCK_MASK) - vm_block;
	struct nvc0_pgt *pt = nvc0_vspace_pgt(vs, pde);

	return ((nv_rv32(pt->bo[s], pte * 8) >> 4) << 12) + offset;
}

int nvc0_vm_read32(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, uint32_t *ptr)
{
	struct drm_device *dev = vs->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t pmem = 0x700000;
	uint64_t phys;
	uint32_t val;

	spin_lock(&dev_priv->pramin_lock);
	phys = nvc0_vm_phys_getaddr(vs, bo, addr);
	if (phys >> 16 != dev_priv->pramin_start) {
		dev_priv->pramin_start = phys >> 16;
		nv_wr32(dev, 0x1700, phys >> 16);
	}
	val = nv_rd32(dev, pmem + (phys & 0xffff));
	spin_unlock(&dev_priv->pramin_lock);

	*ptr = val;

	return 0;
}

int nvc0_vm_write32(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, uint32_t val)
{
	struct drm_device *dev = vs->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t pmem = 0x700000;
	uint64_t phys;

	spin_lock(&dev_priv->pramin_lock);
	phys = nvc0_vm_phys_getaddr(vs, bo, addr);
	if ((phys >> 16) != dev_priv->pramin_start) {
		dev_priv->pramin_start = phys >> 16;
		nv_wr32(dev, 0x1700, phys >> 16);
	}
	nv_wr32(dev, pmem + (phys & 0xffff), val);
	spin_unlock(&dev_priv->pramin_lock);

	return 0;
}

int nvc0_vm_read(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, void *buf, uint32_t size)
{
	struct drm_device *dev = vs->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t pmem = 0x700000;
	uint32_t wsize;
	uint64_t phys;

	do {
		spin_lock(&dev_priv->pramin_lock);
		phys = nvc0_vm_phys_getaddr(vs, bo, addr);
		wsize = PAGE_SIZE;
		if (wsize > size)
			wsize = size;
		dev_priv->pramin_start = phys >> 16;
		nv_wr32(dev, 0x1700, phys >> 16);
		memcpy_fromio(buf, dev_priv->mmio + pmem + (phys & 0xffff), size);
		/*
		 *int i;
		 *for (i = 0; i < wsize / 4; i++)
		 *((uint32_t*)buf)[i] = nv_rd32(dev, pmem + (phys & 0xffff) + i * 4);
		 */
		size -= wsize;
		addr += wsize;
		buf += wsize;
		spin_unlock(&dev_priv->pramin_lock);
	} while (size);

	return 0;
}

int nvc0_vm_write(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, const void *buf, uint32_t size)
{
	struct drm_device *dev = vs->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t pmem = 0x700000;
	uint64_t phys;
	uint32_t wsize;

	do {
		spin_lock(&dev_priv->pramin_lock);
		phys = nvc0_vm_phys_getaddr(vs, bo, addr);
		wsize = PAGE_SIZE;
		if (wsize > size)
			wsize = size;
		dev_priv->pramin_start = phys >> 16;
		nv_wr32(dev, 0x1700, phys >> 16);
		memcpy_toio(dev_priv->mmio + pmem + (phys & 0xffff), buf, wsize);
		/*
		 *int i;
		 *for (i = 0; i < wsize / 4; i++)
		 *    nv_wr32(dev, pmem + (phys & 0xffff) + i * 4, ((uint32_t*)buf)[i]);
		 */
		size -= wsize;
		addr += wsize;
		buf += wsize;
		spin_unlock(&dev_priv->pramin_lock);
	} while (size);

	return 0;
}

int
nvc0_vm_init(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_pgt *pt;
	struct nvc0_vm_engine *vme = kzalloc(sizeof *vme, GFP_KERNEL);
	if (!vme) {
		NV_ERROR(dev, "VM: Couldn't alloc engine\n");
		return -ENOMEM;
	}
	vme->base.takedown = nvc0_vm_takedown;
	vme->base.do_vspace_new = nvc0_vspace_new;
	vme->base.do_vspace_free = nvc0_vspace_free;
	vme->base.place_map = nvc0_vspace_place_map;
	vme->base.do_map = nvc0_vspace_do_map;
	vme->base.do_unmap = nvc0_vspace_do_unmap;
	vme->base.map_user = nvc0_vm_map_user;
	vme->base.map_kernel = nvc0_vm_map_kernel;
	vme->base.bar_flush = nv84_vm_bar_flush;
	vme->base.phys_getaddr = nvc0_vm_phys_getaddr;
	vme->base.read32 = nvc0_vm_read32;
	vme->base.write32 = nvc0_vm_write32;
	vme->base.read = nvc0_vm_read;
	vme->base.write = nvc0_vm_write;
	dev_priv->vm = &vme->base;

	dev_priv->vm_ramin_base = 0;
	spin_lock_init(&dev_priv->vm->vs_lock);

	nv_wr32(dev, 0x200, 0xfffffeff);
	nv_wr32(dev, 0x200, 0xffffffff);

	nv_wr32(dev, 0x100c80, 0x00208000);

	vme->bar3vm = pscnv_vspace_new (dev, dev_priv->ramin_size, 0, 3);
	if (!vme->bar3vm) {
		kfree(vme);
		dev_priv->vm = 0;
		return -ENOMEM;
	}
	vme->bar3ch = pscnv_chan_new (dev, vme->bar3vm, 3);
	if (!vme->bar3ch) {
		pscnv_vspace_unref(vme->bar3vm);
		kfree(vme);
		dev_priv->vm = 0;
		return -ENOMEM;
	}
	nv_wr32(dev, 0x1714, 0xc0000000 | vme->bar3ch->bo->start >> 12);

	dev_priv->vm_ok = 1;

	nvc0_vm_map_kernel(vme->bar3ch->bo);
	nvc0_vm_map_kernel(nvc0_vs(vme->bar3vm)->pd);
	pt = nvc0_vspace_pgt(vme->bar3vm, 0);
	if (!pt) {
		NV_ERROR(dev, "VM: failed to allocate RAMIN page table\n");
		return -ENOMEM;
	}
	nvc0_vm_map_kernel(pt->bo[1]);

	vme->bar1vm = pscnv_vspace_new (dev, dev_priv->fb_size, 0, 1);
	if (!vme->bar1vm) {
		dev_priv->vm_ok = 0;
		pscnv_chan_unref(vme->bar3ch);
		pscnv_vspace_unref(vme->bar3vm);
		kfree(vme);
		dev_priv->vm = 0;
		return -ENOMEM;
	}
	vme->bar1ch = pscnv_chan_new (dev, vme->bar1vm, 1);
	if (!vme->bar1ch) {
		dev_priv->vm_ok = 0;
		pscnv_vspace_unref(vme->bar1vm);
		pscnv_chan_unref(vme->bar3ch);
		pscnv_vspace_unref(vme->bar3vm);
		kfree(vme);
		dev_priv->vm = 0;
		return -ENOMEM;
	}
	nv_wr32(dev, 0x1704, 0x80000000 | vme->bar1ch->bo->start >> 12);
	return 0;
}

void
nvc0_vm_takedown(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nvc0_vm_engine *vme = nvc0_vm(dev_priv->vm);
	/* XXX: write me. */
	dev_priv->vm_ok = 0;
	nv_wr32(dev, 0x1704, 0);
	nv_wr32(dev, 0x1714, 0);
	nv_wr32(dev, 0x1718, 0);
	pscnv_chan_unref(vme->bar1ch);
	pscnv_vspace_unref(vme->bar1vm);
	pscnv_chan_unref(vme->bar3ch);
	pscnv_vspace_unref(vme->bar3vm);
	kfree(vme);
	dev_priv->vm = 0;
}

