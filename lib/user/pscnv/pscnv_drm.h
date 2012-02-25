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

#ifndef __PSCNV_DRM_H__
#define __PSCNV_DRM_H__

#define PSCNV_DRM_HEADER_PATCHLEVEL 1

#define PSCNV_GETPARAM_PCI_VENDOR      3
#define PSCNV_GETPARAM_PCI_DEVICE      4
#define PSCNV_GETPARAM_BUS_TYPE        5
#define PSCNV_GETPARAM_FB_SIZE         8
#define PSCNV_GETPARAM_AGP_SIZE        9
#define PSCNV_GETPARAM_CHIPSET_ID      11
#define PSCNV_GETPARAM_GRAPH_UNITS     13
#define PSCNV_GETPARAM_PTIMER_TIME     14
struct drm_pscnv_getparam {
	uint64_t param;		/* < */
	uint64_t value;		/* > */
};

enum pscnv_bus_type {
	NV_AGP     = 0,
	NV_PCI     = 1,
	NV_PCIE    = 2,
};

/* used for gem_new and gem_info */
struct drm_pscnv_gem_info {	/* n i */
	/* GEM handle used for identification */
	uint32_t handle;	/* > < */
	/* cookie: free-form 32-bit number displayed in debug info. */
	uint32_t cookie;	/* < > */
	/* misc flags, see below. */
	uint32_t flags;		/* < > */
	uint32_t tile_flags;	/* < > */
	uint64_t size;		/* < > */
	/* offset inside drm fd's vm space usable for mmapping */
	uint64_t map_handle;	/* > > */
	/* unused by kernel, can be used by userspace to store some info,
	 * like buffer format and tile_mode for DRI2 */
	uint32_t user[8];	/* < > */
};
#define PSCNV_GEM_CONTIG		0x00000001	/* needs to be contiguous in VRAM */
#define PSCNV_GEM_MAPPABLE		0x00000002	/* intended to be mmapped by host */
#define PSCNV_GEM_MEMTYPE_MASK		0x0000000c
#define PSCNV_GEM_VRAM_SMALL		0x00000000	/* VRAM with small pages */
#define PSCNV_GEM_SYSRAM_SNOOP		0x00000004
#define PSCNV_GEM_VRAM_LARGE		0x00000008	/* VRAM with large pages */
#define PSCNV_GEM_SYSRAM_NOSNOOP	0x0000000c
#define PSCNV_GEM_GART			PSCNV_GEM_SYSRAM_SNOOP	/* compat */

/* for vspace_new and vspace_free */
struct drm_pscnv_vspace_req {	/* n f */
	uint32_t vid;		/* > < */
};

struct drm_pscnv_vspace_map {
	uint32_t vid;		/* < */
	uint32_t handle;	/* < */
	uint64_t start;		/* < */
	uint64_t end;		/* < */
	uint32_t back;		/* < */
	/* none defined yet */
	uint32_t flags;		/* < */
	uint64_t offset;	/* > */
};

struct drm_pscnv_vspace_unmap {
	uint32_t vid;		/* < */
	uint32_t _pad;
	uint64_t offset;	/* < */
};

struct drm_pscnv_chan_new {
	uint32_t vid;		/* < */
	uint32_t cid;		/* > */
	/* The map handle that can be used to access channel control regs */
	uint64_t map_handle;	/* > */
};

struct drm_pscnv_chan_free {
	uint32_t cid;		/* < */
};

struct drm_pscnv_obj_vdma_new {
	uint32_t cid;		/* < */
	uint32_t handle;	/* < */
	uint32_t oclass;	/* < */
	uint32_t flags;		/* < */
	uint64_t start;		/* < */
	uint64_t size;		/* < */
};

struct drm_pscnv_fifo_init {
	uint32_t cid;		/* < */
	uint32_t pb_handle;	/* < */
	uint32_t flags;		/* < */
	uint32_t slimask;	/* < */
	uint64_t pb_start;	/* < */
};

struct drm_pscnv_fifo_init_ib {
	uint32_t cid;		/* < */
	uint32_t pb_handle;	/* < */
	uint32_t flags;		/* < */
	uint32_t slimask;	/* < */
	uint64_t ib_start;	/* < */
	uint32_t ib_order;	/* < */
	uint32_t _pad;
};

struct drm_pscnv_obj_eng_new {
	uint32_t cid;		/* < */
	uint32_t handle;	/* < */
	uint32_t oclass;	/* < */
	uint32_t flags;		/* < */
};

struct drm_pscnv_vm_rw32 {
	uint32_t vid;		/* < */
	uint32_t handle;	/* < */
	uint64_t addr;      /* < */
	uint32_t val;       /* > < */
};

struct drm_pscnv_vm_rw {
	uint32_t vid;		/* < */
	uint32_t handle;	/* < */
	uint64_t addr;      /* < */
	uint32_t size;      /* < */
	void *buf_rd;       /* < */
	const void *buf_wr; /* < */
};

struct drm_pscnv_vm_map {
	uint32_t vid;         /* < */
	uint32_t handle;      /* < */
	uint64_t map_handle;  /* > < */
};

struct drm_pscnv_phys_getaddr {
	uint32_t vid;		 /* < */
	uint32_t handle;	 /* < */
	uint64_t addr;       /* < */
	uint32_t offset;     /* < */
	uint64_t phys;       /* > */
};

#define DRM_PSCNV_GETPARAM           0x00	/* get some information from the card */
#define DRM_PSCNV_GEM_NEW            0x20	/* create a new BO */
#define DRM_PSCNV_GEM_INFO           0x21	/* get info about a BO */
/* also uses generic GEM close, flink, open ioctls */
#define DRM_PSCNV_VSPACE_NEW         0x22	/* Create a new virtual address space */
#define DRM_PSCNV_VSPACE_FREE        0x23	/* Free a virtual address space */
#define DRM_PSCNV_VSPACE_MAP         0x24	/* Maps a BO to a vspace */
#define DRM_PSCNV_VSPACE_UNMAP       0x25	/* Unmaps a BO from a vspace */
#define DRM_PSCNV_CHAN_NEW           0x26	/* Create a new channel */
#define DRM_PSCNV_CHAN_FREE          0x27	/* Free a channel */
#define DRM_PSCNV_OBJ_VDMA_NEW       0x28	/* Create a new vspace DMA object on a channel */
#define DRM_PSCNV_FIFO_INIT          0x29	/* Initialises PFIFO processing on a channel */
#define DRM_PSCNV_OBJ_ENG_NEW        0x2a	/* Create a new engine object on a channel */
#define DRM_PSCNV_FIFO_INIT_IB       0x2b	/* Initialises IB PFIFO processing on a channel */
#define DRM_PSCNV_VM_READ32          0x2c	/* Read from virtual memory */
#define DRM_PSCNV_VM_WRITE32         0x2d	/* Write to virtual memory */
#define DRM_PSCNV_VM_READ            0x2e	/* Read from virtual memory */
#define DRM_PSCNV_VM_WRITE           0x2f	/* Write to virtual memory */
#define DRM_PSCNV_VM_MAP             0x30	/* Map virtual memory */
#define DRM_PSCNV_VM_UNMAP           0x31	/* Unmap virtual memory */
#define DRM_PSCNV_PHYS_GETADDR       0x32	/* Get physical address */

#endif /* __PSCNV_DRM_H__ */
