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

int nv50_vram_alloc(struct pscnv_bo *bo);
int nv50_sysram_tiling_ok(struct pscnv_bo *bo);

int
nv50_vram_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	uint32_t r0, r4, rc, ru, rt;
	int parts, i, colbits, rowbitsa, rowbitsb, banks;
	uint64_t rowsize, predicted;
	uint32_t rblock_size;
	int ret;
	u32 pfb714 = nv_rd32(dev, 0x100714);

	switch (pfb714 & 0x00000007) {
	case 0: dev_priv->vram_type = NV_MEM_TYPE_DDR1; break;
	case 1:
		if (nouveau_mem_vbios_type(dev) == NV_MEM_TYPE_DDR3)
			dev_priv->vram_type = NV_MEM_TYPE_DDR3;
		else
			dev_priv->vram_type = NV_MEM_TYPE_DDR2;
		break;
	case 2: dev_priv->vram_type = NV_MEM_TYPE_GDDR3; break;
	case 3: dev_priv->vram_type = NV_MEM_TYPE_GDDR4; break;
	case 4: dev_priv->vram_type = NV_MEM_TYPE_GDDR5; break;
	default:
		break;
	}
	dev_priv->vram_rank_B = !!(nv_rd32(dev, 0x100200) & 0x4);

	dev_priv->vram = kzalloc (sizeof *dev_priv->vram, GFP_KERNEL);
	if (!dev_priv->vram) {
		NV_ERROR(dev, "VRAM: out ot memory\n");
		return -ENOMEM;
	}

	dev_priv->vram->alloc = nv50_vram_alloc;
	dev_priv->vram->free = pscnv_vram_free;
	dev_priv->vram->takedown = pscnv_vram_takedown;
	dev_priv->vram->sysram_tiling_ok = nv50_sysram_tiling_ok;

	if (dev_priv->chipset == 0xaa || dev_priv->chipset == 0xac || dev_priv->chipset == 0xaf) {
		rc = nv_rd32(dev, 0x10020c);
		dev_priv->vram_sys_base = (uint64_t)nv_rd32(dev, 0x100e10) << 12;
		dev_priv->vram_size = (rc & 0xfffff000) | ((uint64_t)rc & 0xff) << 32;
		rblock_size = 0x1000;

		NV_INFO(dev, "VRAM: IGP stolen area at %llx size 0x%llx",
				dev_priv->vram_sys_base, dev_priv->vram_size);
	} else {
		dev_priv->vram_sys_base = 0;

		r0 = nv_rd32(dev, 0x100200);
		r4 = nv_rd32(dev, 0x100204);
		rc = nv_rd32(dev, 0x10020c);
		rt = nv_rd32(dev, 0x100250);
		ru = nv_rd32(dev, 0x1540);
		NV_INFO (dev, "Memory config regs: %08x %08x %08x %08x %08x\n", r0, r4, rc, rt, ru);

		parts = 0;
		for (i = 0; i < 8; i++)
			if (ru & (1 << (i + 16)))
				parts++;
		colbits = (r4 >> 12) & 0xf;
		rowbitsa = ((r4 >> 16) & 0xf) + 8;
		rowbitsb = ((r4 >> 20) & 0xf) + 8;
		banks = ((r4 & 1 << 24) ? 8 : 4);
		rowsize = parts * banks * (1 << colbits) * 8;
		predicted = rowsize << rowbitsa;
		if (r0 & 4)
			predicted += rowsize << rowbitsb;

		dev_priv->vram_size = (rc & 0xfffff000) | ((uint64_t)rc & 0xff) << 32;
		if (!dev_priv->vram_size) {
			NV_ERROR(dev, "Memory controller claims 0 VRAM - aborting.\n");
			kfree(dev_priv->vram);
			return -ENODEV;
		}
		if (dev_priv->vram_size != predicted) {
			NV_WARN(dev, "Memory controller reports VRAM size of 0x%llx, inconsistent with our calculation of 0x%llx!\n", dev_priv->vram_size, predicted);
		}

		/* XXX: 100250 has more bits. check what they do some day. */
		if (rt & 1)
			rblock_size = rowsize * 3;
		else
			rblock_size = rowsize;

		NV_INFO(dev, "VRAM: size 0x%llx, LSR period %x\n",
				dev_priv->vram_size, rblock_size);
	}

	ret = pscnv_mm_init(dev, 0x40000, dev_priv->vram_size - 0x20000, 0x1000, 0x10000, rblock_size, &dev_priv->vram_mm);
	if (ret) {
		kfree(dev_priv->vram);
		return ret;
	}

	return 0;
}

int
nv50_sysram_tiling_ok(struct pscnv_bo *bo) {
	switch (bo->tile_flags) {
		case 0:
		case 0x10:
		case 0x11:
		case 0x12:
		case 0x13:
		case 0x20:
		case 0x21:
		case 0x22:
		case 0x23:
		case 0x24:
		case 0x25:
		case 0x26:
		case 0x40:
		case 0x41:
		case 0x42:
		case 0x43:
		case 0x44:
		case 0x45:
		case 0x46:
		case 0x54:
		case 0x55:
		case 0x56:
		case 0x60:
		case 0x61:
		case 0x62:
		case 0x63:
		case 0x64:
		case 0x65:
		case 0x66:
		case 0x68:
		case 0x69:
		case 0x6a:
		case 0x6b:
		case 0x70:
		case 0x74:
		case 0x78:
		case 0x79:
		case 0x7c:
		case 0x7d:
		case 0x18:
		case 0x19:
		case 0x1a:
		case 0x1b:
		case 0x28:
		case 0x29:
		case 0x2a:
		case 0x2b:
		case 0x2c:
		case 0x2d:
		case 0x2e:
		case 0x47:
		case 0x48:
		case 0x49:
		case 0x4a:
		case 0x4b:
		case 0x4c:
		case 0x4d:
		case 0x6c:
		case 0x6d:
		case 0x6e:
		case 0x6f:
		case 0x72:
		case 0x76:
		case 0x7a:
		case 0x7b:
			return 1;
		default:
			return 0;
	}
}

int
nv50_vram_alloc(struct pscnv_bo *bo)
{
	struct drm_device *dev = bo->dev;
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	int flags, ret;
	switch (bo->tile_flags) {
		case 0:
		case 0x10:
		case 0x11:
		case 0x12:
		case 0x13:
		case 0x20:
		case 0x21:
		case 0x22:
		case 0x23:
		case 0x24:
		case 0x25:
		case 0x26:
		case 0x40:
		case 0x41:
		case 0x42:
		case 0x43:
		case 0x44:
		case 0x45:
		case 0x46:
		case 0x54:
		case 0x55:
		case 0x56:
		case 0x60:
		case 0x61:
		case 0x62:
		case 0x63:
		case 0x64:
		case 0x65:
		case 0x66:
		case 0x68:
		case 0x69:
		case 0x6a:
		case 0x6b:
		case 0x70:
		case 0x74:
		case 0x78:
		case 0x79:
		case 0x7c:
		case 0x7d:
			flags = 0;
			break;
		case 0x18:
		case 0x19:
		case 0x1a:
		case 0x1b:
		case 0x28:
		case 0x29:
		case 0x2a:
		case 0x2b:
		case 0x2c:
		case 0x2d:
		case 0x2e:
		case 0x47:
		case 0x48:
		case 0x49:
		case 0x4a:
		case 0x4b:
		case 0x4c:
		case 0x4d:
		case 0x6c:
		case 0x6d:
		case 0x6e:
		case 0x6f:
		case 0x72:
		case 0x76:
		case 0x7a:
		case 0x7b:
			flags = PSCNV_MM_T1 | PSCNV_MM_FROMBACK;
			break;
		default:
			return -EINVAL;
	}
	bo->size = roundup(bo->size, 0x1000);
	if ((bo->flags & PSCNV_GEM_MEMTYPE_MASK) == PSCNV_GEM_VRAM_LARGE) {
		flags |= PSCNV_MM_LP;
		bo->size = roundup(bo->size, 0x10000);
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
