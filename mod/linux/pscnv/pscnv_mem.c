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

#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "pscnv_mem.h"
#include "pscnv_vm.h"
#include <linux/list.h>
#include <linux/kernel.h>
#include <linux/mutex.h>

int
pscnv_mem_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	int ret, dma_bits = 32;
	spin_lock_init(&dev_priv->pramin_lock);

	if (dev_priv->card_type >= NV_50 &&
	    pci_dma_supported(dev->pdev, DMA_BIT_MASK(40)))
		dma_bits = 40;

	ret = pci_set_dma_mask(dev->pdev, DMA_BIT_MASK(dma_bits));
	if (ret) {
		NV_ERROR(dev, "Error setting DMA mask: %d\n", ret);
		return ret;
	}

	mutex_init(&dev_priv->vram_mutex);
	
	switch (dev_priv->card_type) {
		case NV_50:
			ret = nv50_vram_init(dev);
			break;
		case NV_C0:
			ret = nvc0_vram_init(dev);
			break;
		default:
			NV_ERROR(dev, "No VRAM allocator for NV%02x!\n", dev_priv->chipset);
			ret = -ENOSYS;
	}
	if (ret)
		return ret;

	dev_priv->fb_mtrr = drm_mtrr_add(pci_resource_start(dev->pdev, 1),
					 pci_resource_len(dev->pdev, 1),
					 DRM_MTRR_WC);

	return 0;
}

void
pscnv_mem_takedown(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	dev_priv->vram->takedown(dev);

	if (dev_priv->fb_mtrr >= 0) {
		drm_mtrr_del(dev_priv->fb_mtrr, pci_resource_start(dev->pdev, 1),
			     pci_resource_len(dev->pdev, 1), DRM_MTRR_WC);
		dev_priv->fb_mtrr = 0;
	}
}

struct pscnv_bo *
pscnv_mem_alloc(struct drm_device *dev,
		uint64_t size, int flags, int tile_flags, uint32_t cookie)
{
	static int serial = 0;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct pscnv_bo *res;
	int ret;
	/* avoid all sorts of integer overflows possible otherwise. */
	if (size >= (1ULL << 40))
		return 0;
	if (!size)
		return 0;

	res = kzalloc (sizeof *res, GFP_KERNEL);
	if (!res)
		return 0;
	size = ALIGN(size, PSCNV_MEM_PAGE_SIZE);
	size = ALIGN(size, PAGE_SIZE);
	res->dev = dev;
	res->size = size;
	res->flags = flags;
	res->tile_flags = tile_flags;
	res->cookie = cookie;
	res->gem = 0;

	/* XXX: another mutex? */
	mutex_lock(&dev_priv->vram_mutex);
	res->serial = serial++;
	mutex_unlock(&dev_priv->vram_mutex);

	if (pscnv_mem_debug >= 1)
		NV_INFO(dev, "Allocating %d, %#llx-byte %sBO of type %08x, tile_flags %x\n", res->serial, size,
				(flags & PSCNV_GEM_CONTIG ? "contig " : ""), cookie, tile_flags);
	switch (res->flags & PSCNV_GEM_MEMTYPE_MASK) {
		case PSCNV_GEM_VRAM_SMALL:
		case PSCNV_GEM_VRAM_LARGE:
			ret = dev_priv->vram->alloc(res);
			break;
		case PSCNV_GEM_SYSRAM_SNOOP:
		case PSCNV_GEM_SYSRAM_NOSNOOP:
			if (dev_priv->vram->sysram_tiling_ok(res))
				ret = pscnv_sysram_alloc(res);
			else
				ret = -EINVAL;
			break;
		default:
			ret = -ENOSYS;
	}
	if (ret) {
		kfree(res);
		return 0;
	}
	return res;
}

int
pscnv_mem_free(struct pscnv_bo *bo)
{
	struct drm_nouveau_private *dev_priv = bo->dev->dev_private;
	if (pscnv_mem_debug >= 1)
		NV_INFO(bo->dev, "Freeing %d, %#llx-byte %sBO of type %08x, tile_flags %x\n", bo->serial, bo->size,
				(bo->flags & PSCNV_GEM_CONTIG ? "contig " : ""), bo->cookie, bo->tile_flags);
	if (dev_priv->vm_ok && bo->map1)
		pscnv_vspace_unmap_node(bo->map1);
	if (dev_priv->vm_ok && bo->map3)
		pscnv_vspace_unmap_node(bo->map3);
	switch (bo->flags & PSCNV_GEM_MEMTYPE_MASK) {
		case PSCNV_GEM_VRAM_SMALL:
		case PSCNV_GEM_VRAM_LARGE:
			dev_priv->vram->free(bo);
			break;
		case PSCNV_GEM_SYSRAM_SNOOP:
		case PSCNV_GEM_SYSRAM_NOSNOOP:
			pscnv_sysram_free(bo);
			break;
	}
	kfree (bo);
	return 0;
}

int
pscnv_vram_free(struct pscnv_bo *bo)
{
	struct drm_nouveau_private *dev_priv = bo->dev->dev_private;
	mutex_lock(&dev_priv->vram_mutex);
	pscnv_mm_free(bo->mmnode);
	mutex_unlock(&dev_priv->vram_mutex);
	return 0;
}

static void pscnv_vram_takedown_free(struct pscnv_mm_node *node) {
	struct pscnv_bo *bo = node->tag;
	NV_ERROR(bo->dev, "BO %d of type %08x still exists at takedown!\n",
			bo->serial, bo->cookie);
	pscnv_mem_free(bo);
}

void
pscnv_vram_takedown(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	pscnv_mm_takedown(dev_priv->vram_mm, pscnv_vram_takedown_free);
}
