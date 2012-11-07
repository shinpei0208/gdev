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
#include "pscnv_ramht.h"
#include "pscnv_vm.h"

uint32_t pscnv_ramht_hash(struct pscnv_ramht *ramht, uint32_t handle) {
	uint32_t hash = 0;
	while (handle) {
		hash ^= handle & ((1 << ramht->bits) - 1);
		handle >>= ramht->bits;
	}
	return hash;
}

int pscnv_ramht_insert(struct pscnv_ramht *ramht, uint32_t handle, uint32_t context) {
	/* XXX: check if the object exists already... */
	struct drm_nouveau_private *dev_priv = ramht->bo->dev->dev_private;
	uint32_t hash = pscnv_ramht_hash(ramht, handle);
	uint32_t start = hash * 8;
	uint32_t pos = start;
	if (pscnv_ramht_debug >= 2)
		NV_INFO(ramht->bo->dev, "Handle %x hash %x\n", handle, hash);
	spin_lock (&ramht->lock);
	do {
		if (!nv_rv32(ramht->bo, ramht->offset + pos + 4)) {
			nv_wv32(ramht->bo, ramht->offset + pos, handle);
			nv_wv32(ramht->bo, ramht->offset + pos + 4, context);
			dev_priv->vm->bar_flush(ramht->bo->dev);
			spin_unlock (&ramht->lock);
			if (pscnv_ramht_debug >= 1)
				NV_INFO(ramht->bo->dev, "Adding RAMHT entry for object %x at %x, context %x\n", handle, pos, context);
			return 0;
		}
		pos += 8;
		if (pos == 8 << ramht->bits)
			pos = 0;
	} while (pos != start);
	spin_unlock (&ramht->lock);
	NV_ERROR(ramht->bo->dev, "No RAMHT space for object %x\n", handle);
	return -ENOMEM;
}

uint32_t pscnv_ramht_find(struct pscnv_ramht *ramht, uint32_t handle) {
	/* XXX: do this properly. */
	uint32_t hash = pscnv_ramht_hash(ramht, handle);
	uint32_t start = hash * 8;
	uint32_t pos = start;
	uint32_t res;
	if (pscnv_ramht_debug >= 2)
		NV_INFO(ramht->bo->dev, "Handle %x hash %x\n", handle, hash);
	spin_lock (&ramht->lock);
	do {
		if (!nv_rv32(ramht->bo, ramht->offset + pos + 4))
			break;
		if (nv_rv32(ramht->bo, ramht->offset + pos) == handle) {
			res = nv_rv32(ramht->bo, ramht->offset + pos + 4);
			spin_unlock (&ramht->lock);
			return res;
		} 
		pos += 8;
		if (pos == 8 << ramht->bits)
			pos = 0;
	} while (pos != start);
	spin_unlock (&ramht->lock);
	NV_ERROR(ramht->bo->dev, "RAMHT object %x not found\n", handle);
	return 0;
}
