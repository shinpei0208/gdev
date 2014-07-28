/*
 * Copyright (C) 2013 Marcin Kościelnicki <koriakin@0x04.net>
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef NVRM_PRIV_H
#define NVRM_PRIV_H

#include "nvrm.h"
#include <inttypes.h>

#define NVRM_MAX_DEV 32

#define NVRM_GPU_ID_INVALID ((uint32_t)0xffffffffu)

struct nvrm_handle {
	struct nvrm_handle *next;
	uint32_t handle;
};

struct nvrm_device {
	struct nvrm_context *ctx;
	int idx;
	int open;
	uint32_t gpu_id;
	int fd;
	uint32_t odev;
	uint32_t osubdev;
};

struct nvrm_context {
	int fd_ctl;
	uint32_t cid;
	struct nvrm_device devs[NVRM_MAX_DEV];
	struct nvrm_handle *hchain;
	int ver_major;
	int ver_minor;
};

struct nvrm_vspace {
	struct nvrm_context *ctx;
	struct nvrm_device *dev;
	uint32_t ovas;
	uint32_t odma;
};

struct nvrm_bo {
	struct nvrm_context *ctx;
	struct nvrm_device *dev;
	struct nvrm_vspace *vas;
	uint32_t handle;
	uint64_t size;
	uint64_t gpu_addr;
	uint64_t foffset;
	void *mmap;
};

struct nvrm_channel {
	struct nvrm_context *ctx;
	struct nvrm_device *dev;
	struct nvrm_vspace *vas;
	struct nvrm_eng *echain;
	uint32_t ofifo;
	void *fifo_mmap;
	uint64_t fifo_foffset;
	uint32_t oedma;
	uint32_t oerr;
	uint32_t cls;
};

struct nvrm_eng {
	struct nvrm_channel *chan;
	struct nvrm_eng *next;
	uint32_t cls;
	uint32_t handle;
};

/* handles */
uint32_t nvrm_handle_alloc(struct nvrm_context *ctx);
void nvrm_handle_free(struct nvrm_context *ctx, uint32_t handle);

/* ioctls */
int nvrm_ioctl_create_vspace(struct nvrm_device *dev, uint32_t parent, uint32_t handle, uint32_t cls, uint32_t flags, uint64_t *limit, uint64_t *foffset);
int nvrm_ioctl_create_dma(struct nvrm_context *ctx, uint32_t parent, uint32_t handle, uint32_t cls, uint32_t flags, uint64_t base, uint64_t limit);
int nvrm_ioctl_call(struct nvrm_context *ctx, uint32_t handle, uint32_t mthd, void *ptr, uint32_t size);
int nvrm_ioctl_create(struct nvrm_context *ctx, uint32_t parent, uint32_t handle, uint32_t cls, void *ptr);
int nvrm_ioctl_destroy(struct nvrm_context *ctx, uint32_t parent, uint32_t handle);
int nvrm_ioctl_unk4d(struct nvrm_context *ctx, uint32_t handle, const char *str);
int nvrm_ioctl_card_info(struct nvrm_context *ctx);
int nvrm_ioctl_get_fb_size(struct nvrm_context *ctx, int idx, uint64_t *size);
int nvrm_ioctl_get_vendor_id(struct nvrm_context *ctx, int idx, uint16_t *id);
int nvrm_ioctl_get_device_id(struct nvrm_context *ctx, int idx, uint16_t *id);
int nvrm_ioctl_env_info(struct nvrm_context *ctx, uint32_t *pat_supported);
int nvrm_ioctl_check_version_str(struct nvrm_context *ctx, uint32_t cmd, const char *vernum);
int nvrm_ioctl_memory(struct nvrm_context *ctx, uint32_t parent, uint32_t vspace, uint32_t handle, uint32_t flags1, uint32_t flags2, uint64_t base, uint64_t size);
int nvrm_ioctl_vspace_map(struct nvrm_context *ctx, uint32_t dev, uint32_t vspace, uint32_t handle, uint64_t base, uint64_t size, uint64_t *addr);
int nvrm_ioctl_vspace_unmap(struct nvrm_context *ctx, uint32_t dev, uint32_t vspace, uint32_t handle, uint64_t addr);
int nvrm_ioctl_host_map(struct nvrm_context *ctx, uint32_t subdev, uint32_t handle, uint64_t base, uint64_t size, uint64_t *foffset);
int nvrm_ioctl_host_unmap(struct nvrm_context *ctx, uint32_t subdev, uint32_t handle, uint64_t foffset);

/* mthds */
int nvrm_mthd_context_list_devices(struct nvrm_context *ctx, uint32_t handle, uint32_t *pciid);
int nvrm_mthd_context_enable_device(struct nvrm_context *ctx, uint32_t handle, uint32_t pciid);
int nvrm_mthd_context_disable_device(struct nvrm_context *ctx, uint32_t handle, uint32_t pciid);

int nvrm_create_cid(struct nvrm_context *ctx);

#endif
