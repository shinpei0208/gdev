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

#ifndef __PSCNV_VM_H__
#define __PSCNV_VM_H__

#include <linux/kref.h>

struct pscnv_bo;
struct pscnv_chan;

struct pscnv_vspace {
	int vid;
	struct drm_device *dev;
	struct mutex lock;
	struct pscnv_mm *mm;
	struct drm_file *filp;
	struct kref ref;
	uint64_t size;
	uint32_t flags;
	void *engdata;
};

struct pscnv_vm_engine {
	void (*takedown) (struct drm_device *dev);
	int (*do_vspace_new) (struct pscnv_vspace *vs);
	void (*do_vspace_free) (struct pscnv_vspace *vs);
	int (*place_map) (struct pscnv_vspace *, struct pscnv_bo *, uint64_t start, uint64_t end, int back, struct pscnv_mm_node **res);
	int (*do_map) (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t offset);
	int (*do_unmap) (struct pscnv_vspace *vs, uint64_t offset, uint64_t length);
	int (*map_user) (struct pscnv_bo *);
	int (*map_kernel) (struct pscnv_bo *);
	void (*bar_flush) (struct drm_device *dev);
	uint64_t (*phys_getaddr) (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr);
	int (*read32) (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, uint32_t *ptr);
	int (*write32) (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, uint32_t val);
	int (*read) (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, void *buf, uint32_t size);
	int (*write) (struct pscnv_vspace *vs, struct pscnv_bo *bo, uint64_t addr, const void *buf, uint32_t size);
	struct pscnv_vspace *fake_vspaces[4];
	struct pscnv_vspace *vspaces[128];
	spinlock_t vs_lock;
};

extern struct pscnv_vspace *pscnv_vspace_new(struct drm_device *, uint64_t size, uint32_t flags, int fake);
extern int pscnv_vspace_map(struct pscnv_vspace *, struct pscnv_bo *, uint64_t start, uint64_t end, int back, struct pscnv_mm_node **res);
extern int pscnv_vspace_unmap(struct pscnv_vspace *, uint64_t start);
extern int pscnv_vspace_unmap_node(struct pscnv_mm_node *node);

extern void pscnv_vspace_ref_free(struct kref *ref);

static inline void pscnv_vspace_ref(struct pscnv_vspace *vs) {
	kref_get(&vs->ref);
}

static inline void pscnv_vspace_unref(struct pscnv_vspace *vs) {
	kref_put(&vs->ref, pscnv_vspace_ref_free);
}

extern int pscnv_mmap(struct file *filp, struct vm_area_struct *vma);


int nv50_vm_init(struct drm_device *dev);
int nvc0_vm_init(struct drm_device *dev);

#endif
