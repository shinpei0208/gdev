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

#ifndef __PSCNV_CHAN_H__
#define __PSCNV_CHAN_H__

#include "pscnv_vm.h"
#include "pscnv_ramht.h"
#include "pscnv_engine.h"
#include <linux/kref.h>

struct pscnv_chan {
	struct drm_device *dev;
	int cid;
	/* protected by ch_lock below, used for lookup */
	uint32_t handle;
	struct pscnv_vspace *vspace;
	struct list_head vspace_list;
	struct pscnv_bo *bo;
	spinlock_t instlock;
	int instpos;
	struct pscnv_ramht ramht;
	uint32_t ramfc;
	struct pscnv_bo *cache;
	struct drm_file *filp;
	struct kref ref;
	void *engdata[PSCNV_ENGINES_NUM];
};

struct pscnv_chan_engine {
	void (*takedown) (struct drm_device *dev);
	int (*do_chan_new) (struct pscnv_chan *ch);
	void (*do_chan_free) (struct pscnv_chan *ch);
	struct pscnv_chan *fake_chans[4];
	struct pscnv_chan *chans[128];
	spinlock_t ch_lock;
	int ch_min, ch_max;
};

extern struct pscnv_chan *pscnv_chan_new(struct drm_device *dev, struct pscnv_vspace *, int fake);

extern void pscnv_chan_ref_free(struct kref *ref);

static inline void pscnv_chan_ref(struct pscnv_chan *ch) {
	kref_get(&ch->ref);
}

static inline void pscnv_chan_unref(struct pscnv_chan *ch) {
	kref_put(&ch->ref, pscnv_chan_ref_free);
}

extern int pscnv_chan_mmap(struct file *filp, struct vm_area_struct *vma);
extern int pscnv_chan_handle_lookup(struct drm_device *dev, uint32_t handle);

int nv50_chan_init(struct drm_device *dev);
int nvc0_chan_init(struct drm_device *dev);

#endif
