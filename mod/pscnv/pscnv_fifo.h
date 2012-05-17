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

#ifndef __PSCNV_FIFO_H__
#define __PSCNV_FIFO_H__

struct pscnv_fifo_engine {
	void (*takedown) (struct drm_device *dev);
	int (*chan_init_dma) (struct pscnv_chan *ch, uint32_t pb_handle, uint32_t flags, uint32_t slimask, uint64_t pb_start);
	int (*chan_init_ib) (struct pscnv_chan *ch, uint32_t pb_handle, uint32_t flags, uint32_t slimask, uint64_t ib_start, uint32_t ib_order);
	void (*chan_kill) (struct pscnv_chan *ch);
};

int nv50_fifo_init(struct drm_device *dev);
int nvc0_fifo_init(struct drm_device *dev);

#endif
