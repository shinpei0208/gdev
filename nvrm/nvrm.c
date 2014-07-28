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

#include "nvrm_priv.h"
#include "nvrm_class.h"
#include "nvrm_mthd.h"
#include "nvrm_def.h"
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int nvrm_open_file(const char *fname) {
	int res = open(fname, O_RDWR);
	if (res < 0)
		return res;
	if (fcntl(res, F_SETFD, FD_CLOEXEC) < 0) {
		close(res);
		return -1;
	}
	return res;
}

struct nvrm_context *nvrm_open() {
	int fd;
	struct nvrm_context *res = calloc(sizeof *res, 1);
	if (!res)
		return res;
	res->fd_ctl = nvrm_open_file("/dev/nvidiactl");
	if (res->fd_ctl < 0) {
		free(res);
		return 0;
	}
#if 0
	if (nvrm_ioctl_check_version_str(res, NVRM_CHECK_VERSION_STR_CMD_STRICT, "313.18") != NVRM_CHECK_VERSION_STR_REPLY_RECOGNIZED) {
		close(res->fd_ctl);
		free(res);
		return 0;
	}
	if (nvrm_ioctl_env_info(res, 0)) {
		close(res->fd_ctl);
		free(res);
		return 0;
	}
	if (nvrm_ioctl_card_info(res)) {
		close(res->fd_ctl);
		free(res);
		return 0;
	}
#endif
	if ((fd = open("/sys/module/nvidia/version", O_RDONLY)) >= 0) {
		int ret;
		char buf[256];
		if ((ret = read(fd, buf, sizeof(buf) - 1)) >= 0) {
			buf[ret] = '\0';
			if (ret == 7) {
				sscanf(buf, "%d.%d\n", &res->ver_major, &res->ver_minor);
			} else {
				char temp1[256], temp2[256];
				sscanf(buf, "NVRM version: NVIDIA UNIX %s Kernel Module  %d.%d  %s", temp1, &res->ver_major, &res->ver_minor, temp2);
			}
		}
		close(fd);
	}
	if (nvrm_create_cid(res)) {
		close(res->fd_ctl);
		free(res);
		return 0;
	}
	uint32_t gpu_id[NVRM_MAX_DEV];
	int i;
	if (nvrm_mthd_context_list_devices(res, res->cid, gpu_id)) {
		close(res->fd_ctl);
		free(res);
		return 0;
	}
	for (i = 0; i < NVRM_MAX_DEV; i++) {
		res->devs[i].idx = i;
		res->devs[i].ctx = res;
		res->devs[i].gpu_id = gpu_id[i];
	}
	return res;
}

void nvrm_close(struct nvrm_context *ctx) {
	close(ctx->fd_ctl);
	free(ctx);
}

static int nvrm_xlat_device(struct nvrm_context *ctx, int idx) {
	int i;
	int oidx = idx;
	for (i = 0; i < NVRM_MAX_DEV; i++)
		if (ctx->devs[i].gpu_id != NVRM_GPU_ID_INVALID)
			if (!idx--)
				return i;
	fprintf(stderr, "nvrm: tried accessing OOB device %d\n", oidx);
	abort();
}

int nvrm_num_devices(struct nvrm_context *ctx) {
	int i;
	int cnt = 0;
	for (i = 0; i < NVRM_MAX_DEV; i++)
		if (ctx->devs[i].gpu_id != NVRM_GPU_ID_INVALID)
			cnt++;
	return cnt;
}

struct nvrm_device *nvrm_device_open(struct nvrm_context *ctx, int idx) {
	idx = nvrm_xlat_device(ctx, idx);
	struct nvrm_device *dev = &ctx->devs[idx];
	if (dev->open++) {
		if (!dev->open) {
			fprintf(stderr, "nvrm: open counter overflow\n");
			abort();
		}
		return 0;
	}
	if (nvrm_mthd_context_enable_device(ctx, ctx->cid, dev->gpu_id)) {
		goto out_enable;
	}
	char buf[20];
	snprintf(buf, 20, "/dev/nvidia%d", idx);
	dev->fd = nvrm_open_file(buf);
	if (dev->fd < 0)
		goto out_open;
#if 0
	if (nvrm_ioctl_unk4d(ctx, ctx->cid))
		goto out_unk4d;
#endif
	struct nvrm_create_device arg = {
		.idx = idx,
		.cid = ctx->cid,
	};
	dev->odev = nvrm_handle_alloc(ctx);
	if (nvrm_ioctl_create(ctx, ctx->cid, dev->odev, NVRM_CLASS_DEVICE_0, &arg))
		goto out_dev;
	dev->osubdev = nvrm_handle_alloc(ctx);
	if (nvrm_ioctl_create(ctx, dev->odev, dev->osubdev, NVRM_CLASS_SUBDEVICE_0, 0))
		goto out_subdev;
	return dev;

out_subdev:
	nvrm_handle_free(ctx, dev->osubdev);
	nvrm_ioctl_destroy(ctx, ctx->cid, dev->odev);
out_dev:
	nvrm_handle_free(ctx, dev->odev);
	close(dev->fd);
out_open:
	nvrm_mthd_context_disable_device(ctx, ctx->cid, dev->gpu_id);
out_enable:
	dev->open--;
	return 0;
}

void nvrm_device_close(struct nvrm_device *dev) {
	struct nvrm_context *ctx = dev->ctx;
	int idx = dev->idx;
	if (!dev->open) {
		fprintf(stderr, "nvrm: closing closed device %d\n", idx);
		abort();
	}
	if (--dev->open) {
		return;
	}
	nvrm_ioctl_call(ctx, dev->osubdev, NVRM_MTHD_SUBDEVICE_UNK0146, 0, 0);
	nvrm_ioctl_destroy(ctx, dev->odev, dev->osubdev);
	nvrm_handle_free(ctx, dev->osubdev);
	nvrm_ioctl_destroy(ctx, ctx->cid, dev->odev);
	nvrm_handle_free(ctx, dev->odev);
	close(dev->fd);
	nvrm_mthd_context_disable_device(ctx, ctx->cid, dev->gpu_id);
}
