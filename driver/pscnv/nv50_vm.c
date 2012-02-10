#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "nv50_vm.h"
#include "pscnv_vm.h"
#include "nv50_chan.h"
#include "pscnv_chan.h"

int nv50_vm_map_kernel(struct pscnv_bo *bo);
void nv50_vm_takedown(struct drm_device *dev);
int nv50_vspace_do_unmap (struct pscnv_vspace *vs, uint64_t offset, uint64_t length);

int
nv50_vm_flush(struct drm_device *dev, int unit) {
	nv_wr32(dev, 0x100c80, unit << 16 | 1);
	if (!nouveau_wait_until(dev, 2000000000ULL, 0x100c80, 1, 0)) {
		NV_ERROR(dev, "TLB flush fail on unit %d!\n", unit);
		return -EIO;
	}
	return 0;
}

int nv50_vspace_tlb_flush (struct pscnv_vspace *vs) {
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	int i, ret;
	nv50_vm_flush(vs->dev, 5); /* PFIFO always active */
	for (i = 0; i < PSCNV_ENGINES_NUM; i++) {
		struct pscnv_engine *eng = dev_priv->engines[i];
		if (nv50_vs(vs)->engref[i])
			if ((ret = eng->tlb_flush(eng, vs)))
				return ret;
	}
	return 0;
}

static int
nv50_vspace_fill_pd_slot (struct pscnv_vspace *vs, uint32_t pdenum) {
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	struct list_head *pos;
	int i;
	uint32_t chan_pd;
	nv50_vs(vs)->pt[pdenum] = pscnv_mem_alloc(vs->dev, NV50_VM_SPTE_COUNT * 8, PSCNV_GEM_CONTIG, 0, 0xa9e7ab1e);
	if (!nv50_vs(vs)->pt[pdenum]) {
		return -ENOMEM;
	}

	if (vs->vid != -1)
		nv50_vm_map_kernel(nv50_vs(vs)->pt[pdenum]);

	for (i = 0; i < NV50_VM_SPTE_COUNT; i++)
		nv_wv32(nv50_vs(vs)->pt[pdenum], i * 8, 0);

	if (dev_priv->chipset == 0x50)
		chan_pd = NV50_CHAN_PD;
	else
		chan_pd = NV84_CHAN_PD;

	list_for_each(pos, &nv50_vs(vs)->chan_list) {
		struct pscnv_chan *ch = list_entry(pos, struct pscnv_chan, vspace_list);
		uint64_t pde = nv50_vs(vs)->pt[pdenum]->start | 3;
		nv_wv32(ch->bo, chan_pd + pdenum * 8 + 4, pde >> 32);
		nv_wv32(ch->bo, chan_pd + pdenum * 8, pde);
	}
	return 0;
}

int
nv50_vspace_place_map (struct pscnv_vspace *vs, struct pscnv_bo *bo,
		uint64_t start, uint64_t end, int back,
		struct pscnv_mm_node **res) {
	return pscnv_mm_alloc(vs->mm, bo->size, back?PSCNV_MM_FROMBACK:0, start, end, res);
}

static int nv50_vspace_map_contig_range (struct pscnv_vspace *vs, uint64_t offset, uint64_t pte, uint64_t size, int lp) {
	int ret;
	/* XXX: add LP support */
	BUG_ON(lp);
	while (size) {
		uint32_t pgnum = offset / 0x1000;
		uint32_t pdenum = pgnum / NV50_VM_SPTE_COUNT;
		uint32_t ptenum = pgnum % NV50_VM_SPTE_COUNT;
		int lev = 0;
		int i;
		while (lev < 7 && size >= (0x1000 << (lev + 1)) && !(offset & (1 << (lev + 12))))
			lev++;
		if (!nv50_vs(vs)->pt[pdenum])
			if ((ret = nv50_vspace_fill_pd_slot (vs, pdenum)))
				return ret;
		for (i = 0; i < (1 << lev); i++) {
			nv_wv32(nv50_vs(vs)->pt[pdenum], (ptenum + i) * 8 + 4, pte >> 32);
			nv_wv32(nv50_vs(vs)->pt[pdenum], (ptenum + i) * 8, pte | lev << 7);
			if (pscnv_vm_debug >= 3)
				NV_INFO(vs->dev, "VM: [%08x][%08x] = %016llx\n", pdenum, ptenum + i, pte | lev << 7);
		}
		size -= (0x1000 << lev);
		offset += (0x1000 << lev);
		pte += (0x1000 << lev);
	}
	return 0;
}

int
nv50_vspace_do_map (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t offset) {
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	struct pscnv_mm_node *n;
	int ret, i;
	uint64_t roff = 0;
	switch (bo->flags & PSCNV_GEM_MEMTYPE_MASK) {
		case PSCNV_GEM_VRAM_SMALL:
		case PSCNV_GEM_VRAM_LARGE:
			for (n = bo->mmnode; n; n = n->next) {
				/* XXX: add LP support */
				uint64_t pte = n->start;
				if (dev_priv->chipset == 0xaa || dev_priv->chipset == 0xac || dev_priv->chipset == 0xaf) {
					pte += dev_priv->vram_sys_base;
					pte |= 0x30;
				}
				pte |= (uint64_t)bo->tile_flags << 40;
				pte |= 1; /* present */
				if ((ret = nv50_vspace_map_contig_range(vs, offset + roff, pte, n->size, 0))) {
					nv50_vspace_do_unmap (vs, offset, bo->size);
					return ret;
				}
				roff += n->size;
			}
			break;
		case PSCNV_GEM_SYSRAM_SNOOP:
		case PSCNV_GEM_SYSRAM_NOSNOOP:
			for (i = 0; i < (bo->size >> PAGE_SHIFT); i++) {
				uint64_t pte = bo->dmapages[i];
				pte |= (uint64_t)bo->tile_flags << 40;
				pte |= 1;
				if ((bo->flags & PSCNV_GEM_MEMTYPE_MASK) == PSCNV_GEM_SYSRAM_SNOOP)
					pte |= 0x20;
				else
					pte |= 0x30;
				if ((ret = nv50_vspace_map_contig_range(vs, offset + roff, pte, PAGE_SIZE, 0))) {
					nv50_vspace_do_unmap (vs, offset, bo->size);
					return ret;
				}
				roff += PAGE_SIZE;
			}
			break;
		default:
			return -ENOSYS;
	}
	dev_priv->vm->bar_flush(vs->dev);
	return 0;
}

int
nv50_vspace_do_unmap (struct pscnv_vspace *vs, uint64_t offset, uint64_t length) {
	struct drm_nouveau_private *dev_priv = vs->dev->dev_private;
	while (length) {
		uint32_t pgnum = offset / 0x1000;
		uint32_t pdenum = pgnum / NV50_VM_SPTE_COUNT;
		uint32_t ptenum = pgnum % NV50_VM_SPTE_COUNT;
		if (nv50_vs(vs)->pt[pdenum]) {
			nv_wv32(nv50_vs(vs)->pt[pdenum], ptenum * 8, 0);
		}
		offset += 0x1000;
		length -= 0x1000;
	}
	dev_priv->vm->bar_flush(vs->dev);
	if (vs->vid == -1) {
		return nv50_vm_flush(vs->dev, 6);
	} else {
		nv50_vspace_tlb_flush(vs);
	}
	return 0;
}

int nv50_vspace_new(struct pscnv_vspace *vs) {
	int ret;

	/* XXX: could actually use it some day... */
	if (vs->size > 1ull << 40)
		return -EINVAL;

	vs->engdata = kzalloc(sizeof(struct nv50_vspace), GFP_KERNEL);
	if (!vs->engdata) {
		NV_ERROR(vs->dev, "VM: Couldn't alloc vspace eng\n");
		return -ENOMEM;
	}
	INIT_LIST_HEAD(&nv50_vs(vs)->chan_list);
	ret = pscnv_mm_init(vs->dev, 0, vs->size, 0x1000, 0x10000, 0x20000000, &vs->mm);
	if (ret) 
		kfree(vs->engdata);
	return ret;
}

void nv50_vspace_free(struct pscnv_vspace *vs) {
	int i;
	for (i = 0; i < NV50_VM_PDE_COUNT; i++) {
		if (nv50_vs(vs)->pt[i]) {
			pscnv_mem_free(nv50_vs(vs)->pt[i]);
		}
	}
	kfree(vs->engdata);
}

int nv50_vm_map_user(struct pscnv_bo *bo) {
	struct drm_nouveau_private *dev_priv = bo->dev->dev_private;
	struct nv50_vm_engine *vme = nv50_vm(dev_priv->vm);
	if (bo->map1)
		return 0;
	return pscnv_vspace_map(vme->barvm, bo, 0, dev_priv->fb_size, 0, &bo->map1);
}

int nv50_vm_map_kernel(struct pscnv_bo *bo) {
	struct drm_nouveau_private *dev_priv = bo->dev->dev_private;
	struct nv50_vm_engine *vme = nv50_vm(dev_priv->vm);
	if (bo->map3)
		return 0;
	return pscnv_vspace_map(vme->barvm, bo, dev_priv->fb_size, dev_priv->fb_size + dev_priv->ramin_size, 0, &bo->map3);
}

void
nv50_vm_bar_flush(struct drm_device *dev) {
	nv_wr32(dev, 0x330c, 1);
	if (!nouveau_wait_until(dev, 2000000000ULL, 0x330c, 2, 0)) {
		NV_ERROR(dev, "BAR flush timeout!\n");
	}
}

void
nv84_vm_bar_flush(struct drm_device *dev) {
	nv_wr32(dev, 0x70000, 1);
	if (!nouveau_wait_until(dev, 2000000000ULL, 0x70000, 2, 0)) {
		NV_ERROR(dev, "BAR flush timeout!\n");
	}
}

uint64_t nv50_vm_phys_getaddr(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr)
{
	struct drm_device *dev = vs->dev;
	NV_ERROR(dev, "nv50_vm_phys_getaddr(): Not supported yet!\n");
	return 0;
}

int nv50_vm_read32(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, uint32_t *ptr)
{
	struct drm_device *dev = vs->dev;
	NV_ERROR(dev, "nv50_vm_read32(): Not supported yet!\n");
	return 0;
}

int nv50_vm_write32(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, uint32_t val)
{
	struct drm_device *dev = vs->dev;
	NV_ERROR(dev, "nv50_vm_write32(): Not supported yet!\n");
	return 0;
}

int nv50_vm_read(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, void *buf, uint32_t size)
{
	struct drm_device *dev = vs->dev;
	NV_ERROR(dev, "nv50_vm_read(): Not supported yet!\n");
	return 0;
}

int nv50_vm_write(struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, const void *buf, uint32_t size)
{
	struct drm_device *dev = vs->dev;
	NV_ERROR(dev, "nv50_vm_write(): Not supported yet!\n");
	return 0;
}

int
nv50_vm_init(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	int bar1dma, bar3dma;
	struct nv50_vm_engine *vme = kzalloc(sizeof *vme, GFP_KERNEL);
	if (!vme) {
		NV_ERROR(dev, "VM: Couldn't alloc engine\n");
		return -ENOMEM;
	}
	vme->base.takedown = nv50_vm_takedown;
	vme->base.do_vspace_new = nv50_vspace_new;
	vme->base.do_vspace_free = nv50_vspace_free;
	vme->base.place_map = nv50_vspace_place_map;
	vme->base.do_map = nv50_vspace_do_map;
	vme->base.do_unmap = nv50_vspace_do_unmap;
	vme->base.map_user = nv50_vm_map_user;
	vme->base.map_kernel = nv50_vm_map_kernel;
	if (dev_priv->chipset == 0x50)
		vme->base.bar_flush = nv50_vm_bar_flush;
	else
		vme->base.bar_flush = nv84_vm_bar_flush;
	vme->base.phys_getaddr = nv50_vm_phys_getaddr;
	vme->base.read32 = nv50_vm_read32;
	vme->base.write32 = nv50_vm_write32;
	vme->base.read = nv50_vm_read;
	vme->base.write = nv50_vm_write;
	dev_priv->vm = &vme->base;

	dev_priv->vm_ramin_base = dev_priv->fb_size;
	spin_lock_init(&dev_priv->vm->vs_lock);

	/* This is needed to get meaningful information from 100c90
	 * on traps. No idea what these values mean exactly. */
	switch (dev_priv->chipset) {
	case 0x50:
		nv_wr32(dev, 0x100c90, 0x0707ff);
		break;
	case 0xa3:
	case 0xa5:
	case 0xa8:
	case 0xaf:
		nv_wr32(dev, 0x100c90, 0x0d0fff);
		break;
	default:
		nv_wr32(dev, 0x100c90, 0x1d07ff);
		break;
	}
	vme->barvm = pscnv_vspace_new (dev, dev_priv->fb_size + dev_priv->ramin_size, 0, 1);
	if (!vme->barvm) {
		kfree(vme);
		dev_priv->vm = 0;
		return -ENOMEM;
	}
	vme->barch = pscnv_chan_new (dev, vme->barvm, 1);
	if (!vme->barch) {
		pscnv_vspace_unref(vme->barvm);
		kfree(vme);
		dev_priv->vm = 0;
		return -ENOMEM;
	}
	nv_wr32(dev, 0x1704, 0x40000000 | vme->barch->bo->start >> 12);
	bar1dma = nv50_chan_dmaobj_new(vme->barch, 0x7fc00000, 0, dev_priv->fb_size);
	bar3dma = nv50_chan_dmaobj_new(vme->barch, 0x7fc00000, dev_priv->fb_size, dev_priv->ramin_size);
	nv_wr32(dev, 0x1708, 0x80000000 | bar1dma >> 4);
	nv_wr32(dev, 0x170c, 0x80000000 | bar3dma >> 4);
	dev_priv->vm_ok = 1;
	nv50_vm_map_kernel(vme->barch->bo);
	nv50_vm_map_kernel(nv50_vs(vme->barvm)->pt[0]);
	return 0;
}

void
nv50_vm_takedown(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nv50_vm_engine *vme = nv50_vm(dev_priv->vm);
	/* XXX: write me. */
	dev_priv->vm_ok = 0;
	nv_wr32(dev, 0x1708, 0);
	nv_wr32(dev, 0x170c, 0);
	nv_wr32(dev, 0x1710, 0);
	nv_wr32(dev, 0x1704, 0);
	pscnv_chan_unref(vme->barch);
	pscnv_vspace_unref(vme->barvm);
	kfree(vme);
	dev_priv->vm = 0;
}

/* VM trap handling on NV50 is some kind of a fucking joke.
 *
 * So, there's this little bugger called MMU, which is in PFB area near
 * 0x100c80 and contains registers to flush the TLB caches, and to report
 * VM traps.
 *
 * And you have several units making use of that MMU. The known ones atm
 * include PGRAPH, PFIFO, the BARs, and the PEEPHOLE. Each of these has its
 * own TLBs. And most of them have several subunits, each having a separate
 * MMU access path.
 *
 * Now, if you use an address that is bad in some way, the MMU responds "NO
 * PAGE!!!11!1". And stores the relevant address + unit + channel into
 * 0x100c90 area, where you can read it. However, it does NOT report an
 * interrupt - this is done by the faulting unit.
 *
 * Now, if you get several page faults at once, which is not that uncommon
 * if you fuck up something in your code, all but the first trap is lost.
 * The unit reporting the trap may or may not also store the address on its
 * own.
 *
 * So we report the trap in two pieces. First we go through all the possible
 * faulters and report their status, which may range anywhere from full access
 * info [like TPDMA] to just "oh! a trap!" [like VFETCH]. Then we ask the MMU
 * for whatever trap it remembers. Then the user can look at dmesg and maybe
 * match them using the MMU status field. Which we should decode someday, but
 * meh for now.
 *
 * As for the Holy Grail of Demand Paging - hah. Who the hell knows. Given the
 * fucked up reporting, the only hope lies in getting all individual units to
 * cooperate. BAR accesses quite obviously cannot be demand paged [not a big
 * problem - that's what host page tables are for]. PFIFO accesses all seem
 * restartable just fine. As for PGRAPH... some, like TPDMA, are already dead
 * when they happen, but maybe there's a DEBUG bit somewhere that changes it.
 * Some others, like M2MF, hang on fault, and are therefore promising. But
 * this requires shitloads of RE repeated for every unit. Have fun.
 *
 */

struct pscnv_enumval {
	int value;
	char *name;
	void *data;
};

static struct pscnv_enumval vm_trap_reasons[] = {
	{ 0, "PT_NOT_PRESENT", 0},
	{ 1, "PT_TOO_SHORT", 0 },
	{ 2, "PAGE_NOT_PRESENT", 0 },
	{ 3, "PAGE_SYSTEM_ONLY", 0 },
	{ 4, "PAGE_READ_ONLY", 0 },
	/* 5 never seen */
	{ 6, "NULL_DMAOBJ", 0 },
	{ 7, "WRONG_MEMTYPE", 0 },
	/* 8-0xa never seen */
	{ 0xb, "VRAM_LIMIT", 0 },
	/* 0xc-0xe never seen */
	{ 0xf, "DMAOBJ_LIMIT", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_dispatch_subsubunits[] = {
	{ 0, "GRCTX", 0 },
	{ 1, "NOTIFY", 0 },
	{ 2, "QUERY", 0 },
	{ 3, "COND", 0 },
	{ 4, "M2M_IN", 0 },
	{ 5, "M2M_OUT", 0 },
	{ 6, "M2M_NOTIFY", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_ccache_subsubunits[] = {
	{ 0, "CB", 0 },
	{ 1, "TIC", 0 },
	{ 2, "TSC", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_tprop_subsubunits[] = {
	{ 0, "RT0", 0 },
	{ 1, "RT1", 0 },
	{ 2, "RT2", 0 },
	{ 3, "RT3", 0 },
	{ 4, "RT4", 0 },
	{ 5, "RT5", 0 },
	{ 6, "RT6", 0 },
	{ 7, "RT7", 0 },
	{ 8, "ZETA", 0 },
	{ 9, "LOCAL", 0 },
	{ 0xa, "GLOBAL", 0 },
	{ 0xb, "STACK", 0 },
	{ 0xc, "DST2D", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_pgraph_subunits[] = {
	{ 0, "STRMOUT", 0 },
	{ 3, "DISPATCH", vm_dispatch_subsubunits },
	{ 5, "CCACHE", vm_ccache_subsubunits },
	{ 7, "CLIPID", 0 },
	{ 9, "VFETCH", 0 },
	{ 0xa, "TEXTURE", 0 },
	{ 0xb, "TPROP", vm_tprop_subsubunits },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_crypt_subsubunits[] = {
	{ 0, "CRCTX", 0 },
	{ 1, "SRC", 0 },
	{ 2, "DST", 0 },
	{ 3, "QUERY", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_pcrypt_subunits[] = {
	{ 0xe, "CRYPT", vm_crypt_subsubunits },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_pfifo_subsubunits[] = {
	{ 0, "PUSHBUF", 0 },
	{ 1, "SEMAPHORE", 0 },
	/* 3 seen. also on semaphore. but couldn't reproduce. */
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_pfifo_subunits[] = {
	/* curious. */
	{ 8, "FIFO", vm_pfifo_subsubunits },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_peephole_subunits[] = {
	/* even more curious. */
	{ 4, "WRITE", 0 },
	{ 8, "READ", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_bar_subsubunits[] = {
	{ 0, "FB", 0 },
	{ 1, "IN", 0 },
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_bar_subunits[] = {
	/* even more curious. */
	{ 4, "WRITE", vm_bar_subsubunits },
	{ 8, "READ", vm_bar_subsubunits },
	/* 0xa also seen. some kind of write. */
	{ 0, 0, 0 },
};

static struct pscnv_enumval vm_units[] = {
	{ 0, "PGRAPH", vm_pgraph_subunits },
	{ 1, "PVP", 0 },
	/* 2, 3 never seen */
	{ 4, "PEEPHOLE", vm_peephole_subunits },
	{ 5, "PFIFO", vm_pfifo_subunits },
	{ 6, "BAR", vm_bar_subunits },
	/* 7 never seen */
	{ 8, "PPPP", 0 },
	{ 9, "PBSP", 0 },
	{ 0xa, "PCRYPT", vm_pcrypt_subunits },
	/* 0xb, 0xc never seen */
	{ 0xd, "PCOPY", 0 },
	{ 0xe, "PDAEMON", 0 },
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

void nv50_vm_trap(struct drm_device *dev) {
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t trap[6];
	int i;
	uint32_t idx = nv_rd32(dev, 0x100c90);
	uint32_t s0, s1, s2, s3;
	char reason[50];
	char unit1[50];
	char unit2[50];
	char unit3[50];
	struct pscnv_enumval *ev;
	int chan;
	if (idx & 0x80000000) {
		idx &= 0xffffff;
		for (i = 0; i < 6; i++) {
			nv_wr32(dev, 0x100c90, idx | i << 24);
			trap[i] = nv_rd32(dev, 0x100c94);
		}
		if (dev_priv->chipset < 0xa3 || (dev_priv->chipset >= 0xaa && dev_priv->chipset <= 0xac)) {
			s0 = trap[0] & 0xf;
			s1 = (trap[0] >> 4) & 0xf;
			s2 = (trap[0] >> 8) & 0xf;
			s3 = (trap[0] >> 12) & 0xf;
		} else {
			s0 = trap[0] & 0xff;
			s1 = (trap[0] >> 8) & 0xff;
			s2 = (trap[0] >> 16) & 0xff;
			s3 = (trap[0] >> 24) & 0xff;
		}
		ev = pscnv_enum_find(vm_trap_reasons, s1);
		if (ev)
			snprintf(reason, sizeof(reason), "%s", ev->name);
		else
			snprintf(reason, sizeof(reason), "0x%x", s1);
		ev = pscnv_enum_find(vm_units, s0);
		if (ev)
			snprintf(unit1, sizeof(unit1), "%s", ev->name);
		else
			snprintf(unit1, sizeof(unit1), "0x%x", s0);
		if (ev && (ev = ev->data) && (ev = pscnv_enum_find(ev, s2)))
			snprintf(unit2, sizeof(unit2), "%s", ev->name);
		else
			snprintf(unit2, sizeof(unit2), "0x%x", s2);
		if (ev && (ev = ev->data) && (ev = pscnv_enum_find(ev, s3)))
			snprintf(unit3, sizeof(unit3), "%s", ev->name);
		else
			snprintf(unit3, sizeof(unit3), "0x%x", s3);
		chan = pscnv_chan_handle_lookup(dev, trap[2] << 16 | trap[1]);
		if (chan != 128) {
			NV_INFO(dev, "VM: Trapped %s at %02x%04x%04x ch %d on %s/%s/%s, reason %s\n",
				(trap[5]&0x100?"read":"write"),
				trap[5]&0xff, trap[4]&0xffff,
				trap[3]&0xffff, chan, unit1, unit2, unit3, reason);
		} else {
			NV_INFO(dev, "VM: Trapped %s at %02x%04x%04x UNKNOWN ch %08x on %s/%s/%s, reason %s\n",
				(trap[5]&0x100?"read":"write"),
				trap[5]&0xff, trap[4]&0xffff,
				trap[3]&0xffff, trap[2] << 16 | trap[1], unit1, unit2, unit3, reason);
		}
		nv_wr32(dev, 0x100c90, idx | 0x80000000);
	}
}
