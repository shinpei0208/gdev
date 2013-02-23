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

#ifndef NVRM_MTHD_H
#define NVRM_MTHD_H

#include <inttypes.h>

struct nvrm_mthd_key_value {
	uint32_t key;
	uint32_t value;
};

/* context */

struct nvrm_mthd_context_unk0101 {
	uint32_t unk00;
	uint32_t unk04;
	uint64_t unk08_ptr;
	uint64_t unk10_ptr;
	uint64_t unk18_ptr;
	uint32_t unk20;
	uint32_t unk24;
};
#define NVRM_MTHD_CONTEXT_UNK0101 0x00000101

/* looks exactly like LIST_DEVICES, wtf? */
struct nvrm_mthd_context_unk0201 {
	uint32_t gpu_id[32];
};
#define NVRM_MTHD_CONTEXT_UNK0201 0x00000201

struct nvrm_mthd_context_unk0202 {
	uint32_t gpu_id;
	uint32_t unk04; /* out */
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
	uint32_t unk18;
	uint32_t unk1c_gpu_id; /* out */
	uint32_t unk20;
	uint32_t unk24;
};
#define NVRM_MTHD_CONTEXT_UNK0202 0x00000202

struct nvrm_mthd_context_unk0301 {
	uint32_t unk00[12];
};
#define NVRM_MTHD_CONTEXT_UNK0301 0x00000301

struct nvrm_mthd_context_list_devices {
	uint32_t gpu_id[32];
};
#define NVRM_MTHD_CONTEXT_LIST_DEVICES 0x00000214

struct nvrm_mthd_context_enable_device {
	uint32_t gpu_id;
	uint32_t unk04[32];
};
#define NVRM_MTHD_CONTEXT_ENABLE_DEVICE 0x00000215

struct nvrm_mthd_context_disable_device {
	uint32_t gpu_id;
	uint32_t unk04[32];
};
#define NVRM_MTHD_CONTEXT_DISABLE_DEVICE 0x00000216

/* device */

struct nvrm_mthd_device_unk0201 {
	uint32_t cnt; /* out */
	uint32_t unk04;
	uint64_t ptr; /* XXX */
};
#define NVRM_MTHD_DEVICE_UNK0201 0x00800201

struct nvrm_mthd_device_unk0280 {
	uint32_t unk00; /* out */
};
#define NVRM_MTHD_DEVICE_UNK0280 0x00800280

struct nvrm_mthd_device_unk1102 {
	uint32_t unk00;
	uint32_t unk04;
	uint64_t ptr; /* XXX */
};
#define NVRM_MTHD_DEVICE_UNK1102 0x00801102

struct nvrm_mthd_device_unk1401 {
	uint32_t unk00;
	uint32_t unk04;
	uint64_t ptr; /* XXX */
};
#define NVRM_MTHD_DEVICE_UNK1401 0x00801401

struct nvrm_mthd_device_unk1701 {
	uint32_t unk00;
	uint32_t unk04;
	uint64_t ptr; /* XXX */
};
#define NVRM_MTHD_DEVICE_UNK1701 0x00801701

struct nvrm_mthd_device_unk170d {
	uint32_t unk00;
	uint32_t unk04;
	uint64_t ptr; /* XXX */
	uint64_t unk10;
};
#define NVRM_MTHD_DEVICE_UNK170D 0x0080170d

/* subdevice */

struct nvrm_mthd_subdevice_unk0101 {
	uint32_t unk00;
	uint32_t unk04;
	uint64_t ptr; /* XXX */
};
#define NVRM_MTHD_SUBDEVICE_UNK0101 0x20800101

struct nvrm_mthd_subdevice_get_name {
	uint32_t unk00;
	char name[0x80];
};
#define NVRM_MTHD_SUBDEVICE_GET_NAME 0x20800110

struct nvrm_mthd_subdevice_unk0119 {
	uint32_t unk00;
};
#define NVRM_MTHD_SUBDEVICE_UNK0119 0x20800119

struct nvrm_mthd_subdevice_unk0123 {
	uint32_t cnt;
	uint32_t _pad;
	uint64_t ptr;
};
#define NVRM_MTHD_SUBDEVICE_UNK0123 0x20800123

struct nvrm_mthd_subdevice_get_fifo_classes {
	uint32_t eng;
	uint32_t cnt;
	uint64_t ptr; /* ints */
};
#define NVRM_MTHD_SUBDEVICE_GET_FIFO_CLASSES 0x20800124

struct nvrm_mthd_subdevice_unk0131 {
	uint32_t unk00;
};
#define NVRM_MTHD_SUBDEVICE_UNK0131 0x20800131

struct nvrm_mthd_subdevice_get_gpc_mask {
	uint32_t gpc_mask;
};
#define NVRM_MTHD_SUBDEVICE_GET_GPC_MASK 0x20800137

struct nvrm_mthd_subdevice_get_tp_mask {
	uint32_t gpc_id;
	uint32_t tp_mask;
};
#define NVRM_MTHD_SUBDEVICE_GET_TP_MASK 0x20800138

struct nvrm_mthd_subdevice_get_gpu_id {
	uint32_t gpu_id;
};
#define NVRM_MTHD_SUBDEVICE_GET_GPU_ID 0x20800142

/* no param */
#define NVRM_MTHD_SUBDEVICE_UNK0145 0x20800145

/* no param */
#define NVRM_MTHD_SUBDEVICE_UNK0146 0x20800146

struct nvrm_mthd_subdevice_get_uuid {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t uuid_len;
	char uuid[0x100];
};
#define NVRM_MTHD_SUBDEVICE_GET_UUID 0x2080014a

struct nvrm_mthd_subdevice_unk0303 {
	uint32_t handle_unk003e;
};
#define NVRM_MTHD_SUBDEVICE_UNK0303 0x20800303

struct nvrm_mthd_subdevice_unk0512 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10; /* out */
	uint32_t unk14;
	uint64_t ptr;
};
#define NVRM_MTHD_SUBDEVICE_UNK0512 0x20800512

struct nvrm_mthd_subdevice_unk0522 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10; /* out */
	uint32_t unk14;
	uint64_t ptr;
};
#define NVRM_MTHD_SUBDEVICE_UNK0522 0x20800522

struct nvrm_mthd_subdevice_unk1201 {
	/* XXX reads MP+0x9c on NVCF */
	uint32_t cnt;
	uint32_t _pad;
	uint64_t ptr; /* key:value */
};
#define NVRM_MTHD_SUBDEVICE_UNK1201 0x20801201

struct nvrm_mthd_subdevice_unk1301 {
	uint32_t cnt;
	uint32_t _pad;
	uint64_t ptr; /* key:value */
};
#define NVRM_MTHD_SUBDEVICE_UNK1301 0x20801301

struct nvrm_mthd_subdevice_get_chipset {
	uint32_t major;
	uint32_t minor;
	uint32_t stepping;
};
#define NVRM_MTHD_SUBDEVICE_GET_CHIPSET 0x20801701

struct nvrm_mthd_subdevice_get_bus_id {
	uint32_t main_id;
	uint32_t subsystem_id;
	uint32_t stepping;
	uint32_t real_product_id;
};
#define NVRM_MTHD_SUBDEVICE_GET_BUS_ID 0x20801801

struct nvrm_mthd_subdevice_unk1802 {
	uint32_t cnt;
	uint32_t _pad;
	uint64_t ptr; /* key:value */
};
#define NVRM_MTHD_SUBDEVICE_UNK1802 0x20801802

struct nvrm_mthd_subdevice_get_bus_info {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t regs_size_mb;
	uint64_t regs_base;
	uint32_t unk18;
	uint32_t fb_size_mb;
	uint64_t fb_base;
	uint32_t unk28;
	uint32_t ramin_size_mb;
	uint64_t ramin_base;
	uint32_t unk38;
	uint32_t unk3c;
	uint64_t unk40;
	uint64_t unk48;
	uint64_t unk50;
	uint64_t unk58;
	uint64_t unk60;
	uint64_t unk68;
	uint64_t unk70;
	uint64_t unk78;
	uint64_t unk80;
};
#define NVRM_MTHD_SUBDEVICE_GET_BUS_INFO 0x20801803

struct nvrm_mthd_subdevice_unk1806 {
	uint32_t unk00[0xa8/4]; /* out */
};
#define NVRM_MTHD_SUBDEVICE_UNK1806 0x20801806

struct nvrm_mthd_subdevice_unk200a {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_MTHD_SUBDEVICE_UNK200A 0x2080200a

/* FIFO */

struct nvrm_mthd_fifo_ib_object_info {
	uint32_t handle;
	uint32_t name;
	uint32_t hwcls;
	uint32_t eng;
#define NVRM_FIFO_ENG_GRAPH 1
#define NVRM_FIFO_ENG_COPY0 2
};
#define NVRM_MTHD_FIFO_IB_OBJECT_INFO 0x906f0101

/* ??? */

struct nvrm_mthd_unk85b6_unk0201 {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_MTHD_UNK85B6_UNK0201 0x85b60201

struct nvrm_mthd_unk85b6_unk0202 {
	uint8_t unk00;
};
#define NVRM_MTHD_UNK85B6_UNK0202 0x85b60202

#endif
