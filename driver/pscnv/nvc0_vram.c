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
#include "nouveau_pm.h"
#include "pscnv_mem.h"

#define NVC0_MEM_CTRLR_COUNT                                         0x00121c74
#define NVC0_MEM_CTRLR_RAM_AMOUNT                                    0x0010f20c

int nvc0_vram_alloc(struct pscnv_bo *bo);
int nvc0_sysram_tiling_ok(struct pscnv_bo *bo);

int
nvc0_vram_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	int ret;
	uint32_t ctrlr_num, ctrlr_amt;

	dev_priv->vram = kzalloc (sizeof *dev_priv->vram, GFP_KERNEL);
	if (!dev_priv->vram) {
		NV_ERROR(dev, "VRAM: out ot memory\n");
		return -ENOMEM;
	}

	dev_priv->vram_type = nouveau_mem_vbios_type(dev);
	dev_priv->vram->alloc = nvc0_vram_alloc;
	dev_priv->vram->free = pscnv_vram_free;
	dev_priv->vram->takedown = pscnv_vram_takedown;
	dev_priv->vram->sysram_tiling_ok = nvc0_sysram_tiling_ok;

	ctrlr_num = nv_rd32(dev, NVC0_MEM_CTRLR_COUNT);
	ctrlr_amt = nv_rd32(dev, NVC0_MEM_CTRLR_RAM_AMOUNT);

	dev_priv->vram_size = ctrlr_num * (ctrlr_amt << 20);

	if (!dev_priv->vram_size) {
		NV_ERROR(dev, "No VRAM detected, aborting.\n");
		return -ENODEV;
	}

	NV_INFO(dev, "VRAM: size 0x%llx, %d controllers\n",
			dev_priv->vram_size, ctrlr_num);

	ret = pscnv_mm_init(dev, 0x40000, dev_priv->vram_size - 0x20000, 0x1000, 0x20000, 0x1000, &dev_priv->vram_mm);
	if (ret) {
		kfree(dev_priv->vram);
		return ret;
	}

	return 0;
}

int
nvc0_sysram_tiling_ok(struct pscnv_bo *bo) {
	switch (bo->tile_flags) {
		case 0:
		case 0xdb:
		case 0xfe:
			return 1;
		default:
			return 0;
	}
}

int
nvc0_vram_alloc(struct pscnv_bo *bo)
{
	struct drm_device *dev = bo->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	int flags, ret;
	if (bo->tile_flags & 0xffffff00)
		return -EINVAL;
	flags = 0;
	bo->size = roundup(bo->size, 0x1000);
	if ((bo->flags & PSCNV_GEM_MEMTYPE_MASK) == PSCNV_GEM_VRAM_LARGE) {
		flags |= PSCNV_MM_LP;
		bo->size = roundup(bo->size, 0x20000);
	}
	if (!(bo->flags & PSCNV_GEM_CONTIG))
		flags |= PSCNV_MM_FRAGOK;
	mutex_lock(&dev_priv->vram_mutex);
	ret = pscnv_mm_alloc(dev_priv->vram_mm, bo->size, flags, 0, dev_priv->vram_size, &bo->mmnode);
	if (!ret) {
		if (bo->flags & PSCNV_GEM_CONTIG)
			bo->start = bo->mmnode->start;
		bo->mmnode->tag = bo;
	}
	mutex_unlock(&dev_priv->vram_mutex);
	return ret;
}
