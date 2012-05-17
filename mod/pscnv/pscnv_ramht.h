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

#ifndef __PSCNV_RAMHT_H__
#define __PSCNV_RAMHT_H__

struct pscnv_ramht {
	struct pscnv_bo *bo;
	spinlock_t lock;
	uint32_t offset;
	int bits;
};

extern uint32_t pscnv_ramht_hash(struct pscnv_ramht *, uint32_t handle);
extern int pscnv_ramht_insert(struct pscnv_ramht *, uint32_t handle, uint32_t context);
extern uint32_t pscnv_ramht_find(struct pscnv_ramht *, uint32_t handle);

#endif
