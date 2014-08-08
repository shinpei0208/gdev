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

#include "nvrm_ioctl.h"
#include "nvrm_priv.h"
#include "nvrm_class.h"
#include <string.h>

int nvrm_ioctl_create_vspace(struct nvrm_device *dev, uint32_t parent, uint32_t handle, uint32_t cls, uint32_t flags, uint64_t *limit, uint64_t *foffset) {
	struct nvrm_ioctl_create_vspace arg = {
		.cid = dev->ctx->cid,
		.parent = parent,
		.handle = handle,
		.cls = cls,
		.flags = flags,
		.foffset = 0,
		.limit = *limit,
		.status = 0,
	};
	if (ioctl(dev->fd, NVRM_IOCTL_CREATE_VSPACE, &arg) < 0)
		return -1;
	if (foffset)
		*foffset = arg.foffset;
	*limit = arg.limit;
	return arg.status;
}

int nvrm_ioctl_create_dma(struct nvrm_context *ctx, uint32_t parent, uint32_t handle, uint32_t cls, uint32_t flags, uint64_t base, uint64_t limit) {
	struct nvrm_ioctl_create_dma arg = {
		.cid = ctx->cid,
		.handle = handle,
		.cls = cls,
		.flags = flags,
		.unk10 = 0,
		.parent = parent,
		.base = base,
		.limit = limit,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CREATE_DMA, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_call(struct nvrm_context *ctx, uint32_t handle, uint32_t mthd, void *ptr, uint32_t size) {
	struct nvrm_ioctl_call arg = {
		.cid = ctx->cid,
		.handle = handle,
		.mthd = mthd,
		.ptr = (uint64_t)ptr,
		.size = size,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CALL, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_create(struct nvrm_context *ctx, uint32_t parent, uint32_t handle, uint32_t cls, void *ptr) {
	struct nvrm_ioctl_create arg = {
		.cid = ctx->cid,
		.parent = parent,
		.handle = handle,
		.cls = cls,
		.ptr = (uint64_t)ptr,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CREATE, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_destroy(struct nvrm_context *ctx, uint32_t parent, uint32_t handle) {
	struct nvrm_ioctl_destroy arg = {
		.cid = ctx->cid,
		.parent = parent,
		.handle = handle,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_DESTROY, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_query(struct nvrm_context *ctx, uint32_t handle, uint32_t query, void *ptr, uint32_t size) {
	struct nvrm_ioctl_query arg = {
		.cid = ctx->cid,
		.handle = handle,
		.query = query,
		.ptr = (uint64_t)ptr,
		.size = size,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_QUERY, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_unk4d(struct nvrm_context *ctx, uint32_t handle, const char *str) {
	struct nvrm_ioctl_unk4d arg = {
		.cid = ctx->cid,
		.handle = handle,
		.unk08 = 1,
		.unk10 = 0,
		.slen = strlen(str),
		.sptr = (uint64_t)str,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_UNK4D, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_card_info(struct nvrm_context *ctx) {
	struct nvrm_ioctl_card_info2 arg2 = { {
		{ 0xffffffff },
	} };
	struct nvrm_ioctl_card_info *arg = (struct nvrm_ioctl_card_info *)&arg2;
	if (ctx->ver_major < 331) {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO, arg) < 0)
			return -1;
	} else {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO2, arg) < 0)
			return -1;
	}
	return 0;
}

int nvrm_ioctl_get_fb_size(struct nvrm_context *ctx, int idx, uint64_t *size) {
	struct nvrm_ioctl_card_info2 arg2 = { {
		{ 0xffffffff },
	} };
	struct nvrm_ioctl_card_info *arg = (struct nvrm_ioctl_card_info *)&arg2;
	if (ctx->ver_major < 331) {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO, arg) < 0)
			return -1;
		*size = arg->card[idx].fb_size;
	} else {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO2, arg) < 0)
			return -1;
		*size = arg2.card[idx].fb_size;
	}
	return 0;
}

int nvrm_ioctl_get_vendor_id(struct nvrm_context *ctx, int idx, uint16_t *id) {
	struct nvrm_ioctl_card_info2 arg2 = { {
		{ 0xffffffff },
	} };
	struct nvrm_ioctl_card_info *arg = (struct nvrm_ioctl_card_info *)&arg2;
	if (ctx->ver_major < 331) {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO, arg) < 0)
			return -1;
		*id = arg->card[idx].vendor_id;
	} else {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO2, arg) < 0)
			return -1;
		*id = arg2.card[idx].vendor_id;
	}
	return 0;
}

int nvrm_ioctl_get_device_id(struct nvrm_context *ctx, int idx, uint16_t *id) {
	struct nvrm_ioctl_card_info2 arg2 = { {
		{ 0xffffffff },
	} };
	struct nvrm_ioctl_card_info *arg = (struct nvrm_ioctl_card_info *)&arg2;
	if (ctx->ver_major < 331) {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO, arg) < 0)
			return -1;
		*id = arg->card[idx].device_id;
	} else {
		if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CARD_INFO2, arg) < 0)
			return -1;
		*id = arg2.card[idx].device_id;
	}
	return 0;
}

int nvrm_ioctl_env_info(struct nvrm_context *ctx, uint32_t *pat_supported) {
	struct nvrm_ioctl_env_info arg = {
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_ENV_INFO, &arg) < 0)
		return -1;
	if (pat_supported)
		*pat_supported = arg.pat_supported;
	return 0;
}

int nvrm_ioctl_check_version_str(struct nvrm_context *ctx, uint32_t cmd, const char *vernum) {
	struct nvrm_ioctl_check_version_str arg = {
		.cmd = cmd,
		.reply = 0,
	};
	strncpy(arg.vernum, vernum, sizeof arg.vernum);
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_CHECK_VERSION_STR, &arg) < 0)
		return -1;
	return arg.reply;
}

int nvrm_ioctl_vspace_map(struct nvrm_context *ctx, uint32_t dev, uint32_t vspace, uint32_t handle, uint64_t base, uint64_t size, uint64_t *addr) {
	struct nvrm_ioctl_vspace_map arg = {
		.cid = ctx->cid,
		.dev = dev,
		.vspace = vspace,
		.handle = handle,
		.base = base,
		.size = size,
		.flags = 0x00000,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_VSPACE_MAP, &arg) < 0)
		return -1;
	*addr = arg.addr;
	return arg.status;
}

int nvrm_ioctl_vspace_unmap(struct nvrm_context *ctx, uint32_t dev, uint32_t vspace, uint32_t handle, uint64_t addr) {
	struct nvrm_ioctl_vspace_unmap arg = {
		.cid = ctx->cid,
		.dev = dev,
		.vspace = vspace,
		.handle = handle,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_VSPACE_UNMAP, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_host_map(struct nvrm_context *ctx, uint32_t subdev, uint32_t handle, uint64_t base, uint64_t size, uint64_t *foffset) {
	struct nvrm_ioctl_host_map arg = {
		.cid = ctx->cid,
		.subdev = subdev,
		.handle = handle,
		.base = base,
		.limit = size-1,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_HOST_MAP, &arg) < 0)
		return -1;
	*foffset = arg.foffset;
	return arg.status;
}

int nvrm_ioctl_host_unmap(struct nvrm_context *ctx, uint32_t subdev, uint32_t handle, uint64_t foffset) {
	struct nvrm_ioctl_host_unmap arg = {
		.cid = ctx->cid,
		.subdev = subdev,
		.handle = handle,
		.foffset = foffset,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_HOST_UNMAP, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_ioctl_memory(struct nvrm_context *ctx, uint32_t parent, uint32_t vspace, uint32_t handle, uint32_t flags1, uint32_t flags2, uint64_t base, uint64_t size) {
	struct nvrm_ioctl_memory arg = {
		.cid = ctx->cid,
		.parent = parent,
		.cls = 2,
		.vspace = vspace,
		.handle = handle,
		.flags1 = flags1,
		.flags2 = flags2,
		.size = size,
		.base = base,
		.status = 0,
	};
	if (ioctl(ctx->fd_ctl, NVRM_IOCTL_MEMORY, &arg) < 0)
		return -1;
	return arg.status;
}

int nvrm_create_cid(struct nvrm_context *ctx) {
	return nvrm_ioctl_create(ctx, 0, 0, NVRM_CLASS_CONTEXT_NEW, &ctx->cid);
}
