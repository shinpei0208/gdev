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

#ifndef NVRM_H
#define NVRM_H

#include <stdint.h>

struct nvrm_context;
struct nvrm_device;
struct nvrm_vspace;
struct nvrm_bo;
struct nvrm_channel;
struct nvrm_eng;

struct nvrm_context *nvrm_open();
void nvrm_close(struct nvrm_context *ctx);

int nvrm_num_devices(struct nvrm_context *ctx);
struct nvrm_device *nvrm_device_open(struct nvrm_context *ctx, int idx);
void nvrm_device_close(struct nvrm_device *dev);
int nvrm_device_get_chipset(struct nvrm_device *dev, uint32_t *major, uint32_t *minor, uint32_t *stepping);
int nvrm_device_get_fb_size(struct nvrm_device *dev, uint64_t *fb_size);
int nvrm_device_get_vendor_id(struct nvrm_device *dev, uint16_t *vendor_id);
int nvrm_device_get_device_id(struct nvrm_device *dev, uint16_t *device_id);
int nvrm_device_get_gpc_mask(struct nvrm_device *dev, uint32_t *mask);
int nvrm_device_get_gpc_tp_mask(struct nvrm_device *dev, int gpc_id, uint32_t *mask);
int nvrm_device_get_total_tp_count(struct nvrm_device *dev, int *count);

struct nvrm_vspace *nvrm_vspace_create(struct nvrm_device *dev);
void nvrm_vspace_destroy(struct nvrm_vspace *vas);

struct nvrm_bo *nvrm_bo_create(struct nvrm_vspace *vas, uint64_t size, int sysram);
void nvrm_bo_destroy(struct nvrm_bo *bo);
void *nvrm_bo_host_map(struct nvrm_bo *bo);
uint64_t nvrm_bo_gpu_addr(struct nvrm_bo *bo);
void nvrm_bo_host_unmap(struct nvrm_bo *bo);

struct nvrm_channel *nvrm_channel_create_ib(struct nvrm_vspace *vas, uint32_t cls, struct nvrm_bo *ib);
int nvrm_channel_activate(struct nvrm_channel *chan);
void nvrm_channel_destroy(struct nvrm_channel *chan);
void *nvrm_channel_host_map_regs(struct nvrm_channel *chan);
void *nvrm_channel_host_map_errnot(struct nvrm_channel *chan);

struct nvrm_eng *nvrm_eng_create(struct nvrm_channel *chan, uint32_t eid, uint32_t cls);

#endif
