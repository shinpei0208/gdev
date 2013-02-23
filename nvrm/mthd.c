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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "nvrm_priv.h"
#include "nvrm_mthd.h"

int nvrm_mthd_context_list_devices(struct nvrm_context *ctx, uint32_t handle, uint32_t *gpu_id) {
	struct nvrm_mthd_context_list_devices arg = {
	};
	int res = nvrm_ioctl_call(ctx, handle, NVRM_MTHD_CONTEXT_LIST_DEVICES, &arg, sizeof arg);
	int i;
	if (res)
		return res;
	for (i = 0; i < 32; i++)
		gpu_id[i] = arg.gpu_id[i];
	return 0;
}

int nvrm_mthd_context_enable_device(struct nvrm_context *ctx, uint32_t handle, uint32_t gpu_id) {
	struct nvrm_mthd_context_enable_device arg = {
		.gpu_id = gpu_id,
		.unk04 = { 0xffffffff },	
	};
	return nvrm_ioctl_call(ctx, handle, NVRM_MTHD_CONTEXT_ENABLE_DEVICE, &arg, sizeof arg);
}

int nvrm_mthd_context_disable_device(struct nvrm_context *ctx, uint32_t handle, uint32_t gpu_id) {
	struct nvrm_mthd_context_disable_device arg = {
		.gpu_id = gpu_id,
		.unk04 = { 0xffffffff },	
	};
	return nvrm_ioctl_call(ctx, handle, NVRM_MTHD_CONTEXT_DISABLE_DEVICE, &arg, sizeof arg);
}

int nvrm_device_get_chipset(struct nvrm_device *dev, uint32_t *major, uint32_t *minor, uint32_t *stepping) {
	struct nvrm_mthd_subdevice_get_chipset arg;
	int res = nvrm_ioctl_call(dev->ctx, dev->osubdev, NVRM_MTHD_SUBDEVICE_GET_CHIPSET, &arg, sizeof arg);
	if (res)
		return res;
	if (major) *major = arg.major;
	if (minor) *minor = arg.minor;
	if (stepping) *stepping = arg.stepping;
	return 0;
}
