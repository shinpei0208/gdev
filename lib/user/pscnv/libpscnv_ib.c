#include "libpscnv_ib.h"
#include "libpscnv.h"
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <sys/mman.h>

int pscnv_ib_chan_new(int fd, int vid, struct pscnv_ib_chan **res, uint32_t pb_dma, uint32_t pb_order, uint32_t ib_order, uint32_t chipset) {
	int ret;
	struct pscnv_ib_chan *rr;
	uint64_t map_handle;
	*res = malloc(sizeof **res);
	if (!*res)
		return 1;
	rr = *res;
	rr->fd = fd;
	rr->vid = vid;
	if (!vid) {
		ret = pscnv_vspace_new(fd, (uint32_t*)&rr->vid);
		if (ret)
			goto out_vs;
	}
	ret = pscnv_chan_new(fd, (uint32_t)rr->vid, (uint32_t*)&rr->cid, &map_handle);
	if (ret)
	    goto out_chan;

	rr->chmap = mmap(0, 0x1000, PROT_READ | PROT_WRITE,
			 MAP_SHARED, fd, map_handle);

	if ((void*)rr->chmap == MAP_FAILED)
		goto out_chmap;
	rr->pb_dma = pb_dma;

	if (chipset < 0xc0) {
		ret = pscnv_obj_vdma_new(fd, rr->cid, pb_dma, 0x3d, 0, 0, 1ull << 40);
		if (ret)
			goto out_vdma;
	}
	
	rr->ib_order = ib_order;
	if (!ib_order)
		rr->ib_order = 9;
	ret = pscnv_ib_bo_alloc(fd, rr->vid, 0xf1f01b, PSCNV_GEM_SYSRAM_SNOOP | PSCNV_GEM_MAPPABLE, 0, 8 << rr->ib_order, 0, &rr->ib);
	if (ret)
		goto out_ib;
	rr->ib_map = rr->ib->map;
	rr->ib_mask = (1 << rr->ib_order) - 1;
	rr->ib_put = rr->ib_get = 0;
	rr->pb_order = pb_order;
	if (!pb_order)
		rr->pb_order = 20;
	ret = pscnv_ib_bo_alloc(fd, rr->vid, 0xf1f0, PSCNV_GEM_SYSRAM_SNOOP | PSCNV_GEM_MAPPABLE, 0, 1 << rr->pb_order, 0, &rr->pb);
	if (ret)
		goto out_pb;
	rr->pb_map = rr->pb->map;
	rr->pb_base = rr->pb->vm_base;
	rr->pb_mask = (1 << rr->pb_order) - 1;
	rr->pb_size = (1 << rr->pb_order);
	rr->pb_pos = 0;
	rr->pb_put = 0;
	rr->pb_get = 0;
	ret = pscnv_fifo_init_ib(fd, rr->cid, rr->pb_dma, 0, 1, rr->ib->vm_base, rr->ib_order);
	if (ret)
		goto out_fifo;
	return 0;

out_fifo:
	pscnv_ib_bo_free(rr->pb);
out_pb:
	pscnv_ib_bo_free(rr->ib);
out_ib:
out_vdma:
	munmap((void*)rr->chmap, 0x1000);
out_chmap:
	pscnv_chan_free(fd, rr->cid);
out_chan:
	if (!vid)
		pscnv_vspace_free(fd, rr->vid);
out_vs:
	free(rr);
	return 1;
}

int pscnv_ib_bo_alloc(int fd, int vid, uint32_t cookie, uint32_t flags, uint32_t tile_flags, uint64_t size, uint32_t *user, struct pscnv_ib_bo **res) {
	int ret;
	struct pscnv_ib_bo *rr;
	uint64_t map_handle;
	*res = malloc(sizeof **res);
	if (!*res)
		return 1;
	rr = *res;
	rr->fd = fd;
	rr->vid = vid;
	rr->size = size;
	ret = pscnv_gem_new(fd, cookie, flags, tile_flags, size, user, &rr->handle, &map_handle);
	if (ret)
		goto out_new;
	if (vid) {
		ret = pscnv_vspace_map(fd, vid, rr->handle, 0x20000000, 1ull << 40, 0, 0, &rr->vm_base);
		if (ret)
			goto out_vmap;
	}
	if (flags & PSCNV_GEM_MAPPABLE) {
		rr->map = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map_handle);
		if ((void*)rr->map == MAP_FAILED)
			goto out_map;
	} else
		rr->map = 0;
	return 0;

out_map:
	if (vid)
		pscnv_vspace_unmap(fd, vid, rr->vm_base);
out_vmap:
	pscnv_gem_close(fd, rr->handle);
out_new:
	free(rr);
	return 1;
}

int pscnv_ib_bo_free(struct pscnv_ib_bo *bo) {
	if (bo->map)
		munmap(bo->map, bo->size);
	if (bo->vid)
		pscnv_vspace_unmap(bo->fd, bo->vid, bo->vm_base);
	pscnv_gem_close(bo->fd, bo->handle);
	free(bo);
	return 0;
}

int pscnv_ib_update_get(struct pscnv_ib_chan *ch) {
	uint32_t lo = ch->chmap[0x58/4];
	uint32_t hi = ch->chmap[0x5c/4];
	if (hi & 0x80000000) {
		uint64_t mg = ((uint64_t)hi << 32 | lo) & 0xffffffffffull;
		ch->pb_get = mg - ch->pb_base;
	} else {
		ch->pb_get = 0;
	}
	return 0;
}

int pscnv_ib_push(struct pscnv_ib_chan *ch, uint64_t base, uint32_t len, int flags) {
	uint64_t w = base | (uint64_t)len << 40 | (uint64_t)flags << 40;
	while (((ch->ib_put + 1) & ch->ib_mask) == ch->ib_get) {
		uint32_t old = ch->ib_get;
		ch->ib_get = ch->chmap[0x88/4];
		if (old == ch->ib_get)
			sched_yield();
	}
	ch->ib_map[ch->ib_put * 2] = w;
	ch->ib_map[ch->ib_put * 2 + 1] = w >> 32;
	ch->ib_put++;
	ch->ib_put &= ch->ib_mask;
	ch->chmap[0x8c/4] = ch->ib_put;

	return 0;	
}
