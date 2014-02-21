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

#ifndef NVRM_CLASS_H
#define NVRM_CLASS_H

/* sw objects */

struct nvrm_create_context {
	uint32_t cid;
};
#define NVRM_CLASS_CONTEXT_NEW		0x0000
#define NVRM_CLASS_CONTEXT		0x0041

struct nvrm_create_device {
	uint32_t idx;
	uint32_t cid;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
	uint32_t unk18;
	uint32_t unk1c;
};
#define NVRM_CLASS_DEVICE_0		0x0080	/* wrt context */
#define NVRM_CLASS_DEVICE_1		0x0081	/* wrt context */
#define NVRM_CLASS_DEVICE_2		0x0082	/* wrt context */
#define NVRM_CLASS_DEVICE_3		0x0083	/* wrt context */
#define NVRM_CLASS_DEVICE_4		0x0084	/* wrt context */
#define NVRM_CLASS_DEVICE_5		0x0085	/* wrt context */
#define NVRM_CLASS_DEVICE_6		0x0086	/* wrt context */
#define NVRM_CLASS_DEVICE_7		0x0087	/* wrt context */

struct nvrm_create_subdevice {
	uint32_t idx;
};
#define NVRM_CLASS_SUBDEVICE_0		0x2080	/* wrt device */
#define NVRM_CLASS_SUBDEVICE_1		0x2081	/* wrt device */
#define NVRM_CLASS_SUBDEVICE_2		0x2082	/* wrt device */
#define NVRM_CLASS_SUBDEVICE_3		0x2083	/* wrt device */

/* no create param */
#define NVRM_CLASS_TIMER		0x0004	/* wrt subdevice */

/* created by create_memory */
#define NVRM_CLASS_MEMORY_SYSRAM	0x003e	/* wrt device */
#define NVRM_CLASS_MEMORY_UNK003F	0x003f	/* wrt device */
#define NVRM_CLASS_MEMORY_UNK0040	0x0040	/* wrt device */
#define NVRM_CLASS_MEMORY_VM		0x0070	/* wrt device */
#define NVRM_CLASS_MEMORY_UNK0071	0x0071	/* wrt device */

/* no create param */
#define NVRM_CLASS_UNK0073		0x0073	/* wrt device; singleton */

/* no create param */
#define NVRM_CLASS_UNK208F		0x208f	/* wrt subdevice */

struct nvrm_create_event {
	uint32_t cid;
	uint32_t cls;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t ehandle;
	uint32_t unk14;
};
#define NVRM_CLASS_EVENT		0x0079	/* wrt graph, etc. */

#define NVRM_CLASS_DMA_READ		0x0002	/* wrt memory */
#define NVRM_CLASS_DMA_WRITE		0x0003	/* wrt memory */
#define NVRM_CLASS_DMA_RW		0x003d	/* wrt memory */

/* no create param */
#define NVRM_CLASS_PEEPHOLE_NV30	0x307e
#define NVRM_CLASS_PEEPHOLE_GF100	0x9068

struct nvrm_create_unk83de {
	uint32_t unk00;
	uint32_t cid;
	uint32_t handle; /* seen with compute */
};
#define NVRM_CLASS_UNK83DE		0x83de	/* wrt context */

/* no create param */
#define NVRM_CLASS_UNK85B6		0x85b6	/* wrt subdevice */

/* no create param */
#define NVRM_CLASS_UNK9096		0x9096	/* wrt subdevice */

struct nvrm_create_unk0005 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
};
#define NVRM_CLASS_UNK0005		0x0005

/* FIFO */

struct nvrm_create_fifo_pio {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_FIFO_PIO_NV4		0x006d

struct nvrm_create_fifo_dma {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
	uint32_t unk18;
	uint32_t unk1c;
};
#define NVRM_CLASS_FIFO_DMA_NV40	0x406e
#define NVRM_CLASS_FIFO_DMA_NV44	0x446e

struct nvrm_create_fifo_ib {
	uint32_t error_notify;
	uint32_t dma;
	uint64_t ib_addr;
	uint64_t ib_entries;
	uint32_t unk18;
	uint32_t unk1c;
};
#define NVRM_CLASS_FIFO_IB_G80		0x506f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_G82		0x826f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_MCP89	0x866f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_GF100	0x906f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_GK104	0xa06f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_GK110	0xa16f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_UNKA2	0xa26f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_UNKB0	0xb06f	/* wrt device */

/* graph */

struct nvrm_create_graph {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
};
#define NVRM_CLASS_GR_NULL		0x0030

#define NVRM_CLASS_GR_BETA_NV1		0x0012
#define NVRM_CLASS_GR_CLIP_NV1		0x0019
#define NVRM_CLASS_GR_ROP_NV3		0x0043
#define NVRM_CLASS_GR_BETA4_NV4		0x0072
#define NVRM_CLASS_GR_CHROMA_NV4	0x0057
#define NVRM_CLASS_GR_PATTERN_NV4	0x0044
#define NVRM_CLASS_GR_GDI_NV4		0x004a
#define NVRM_CLASS_GR_BLIT_NV4		0x005f
#define NVRM_CLASS_GR_TRI_NV4		0x005d
#define NVRM_CLASS_GR_IFC_NV30		0x308a
#define NVRM_CLASS_GR_IIFC_NV30		0x3064
#define NVRM_CLASS_GR_LIN_NV30		0x305c
#define NVRM_CLASS_GR_SIFC_NV30		0x3066
#define NVRM_CLASS_GR_TEX_NV30		0x307b
#define NVRM_CLASS_GR_SURF2D_G80	0x5062
#define NVRM_CLASS_GR_SIFM_G80		0x5089
/* XXX: 0052, 0062, 0077, 007b, 009e, 309e still exist??? */

#define NVRM_CLASS_GR_2D_G80		0x502d
#define NVRM_CLASS_GR_M2MF_G80		0x5039
#define NVRM_CLASS_GR_3D_G80		0x5097
#define NVRM_CLASS_GR_COMPUTE_G80	0x50c0

#define NVRM_CLASS_GR_3D_G82		0x8297
#define NVRM_CLASS_GR_3D_G200		0x8397
#define NVRM_CLASS_GR_3D_GT212		0x8597
#define NVRM_CLASS_GR_COMPUTE_GT212	0x85c0 /* XXX: no create param for that one?? */
#define NVRM_CLASS_GR_3D_MCP89		0x8697

#define NVRM_CLASS_GR_2D_GF100		0x902d
#define NVRM_CLASS_GR_M2MF_GF100	0x9039
#define NVRM_CLASS_GR_3D_GF100		0x9097
#define NVRM_CLASS_GR_COMPUTE_GF100	0x90c0

#define NVRM_CLASS_GR_COMPUTE_GF110	0x91c0
#define NVRM_CLASS_GR_3D_GF108		0x9197
#define NVRM_CLASS_GR_3D_GF110		0x9297

#define NVRM_CLASS_GR_UPLOAD_GK104	0xa040
#define NVRM_CLASS_GR_3D_GK104		0xa097
#define NVRM_CLASS_GR_COMPUTE_GK104	0xa0c0

#define NVRM_CLASS_GR_UPLOAD_GK110	0xa140
#define NVRM_CLASS_GR_3D_GK110		0xa197
#define NVRM_CLASS_GR_COMPUTE_GK110	0xa1c0

#define NVRM_CLASS_GR_3D_GK208		0xa297

#define NVRM_CLASS_GR_3D_GM107		0xb097
#define NVRM_CLASS_GR_COMPUTE_GM107	0xb0c0

#define NVRM_CLASS_GR_3D_UNKB1		0xb197

/* copy */

struct nvrm_create_copy {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_COPY_GT212		0x85b5
#define NVRM_CLASS_COPY_GF100_0		0x90b5
#define NVRM_CLASS_COPY_GF100_1		0x90b8 /* XXX: wtf? */
#define NVRM_CLASS_COPY_GK104		0xa0b5
#define NVRM_CLASS_COPY_GM107		0xb0b5

/* vdec etc. */

/* no create param */
#define NVRM_CLASS_MPEG_NV31		0x3174
#define NVRM_CLASS_MPEG_G82		0x8274

struct nvrm_create_me {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_ME_NV40		0x4075

struct nvrm_create_vp {
	uint32_t unk00[0x50/4];
};
#define NVRM_CLASS_VP_G80		0x5076
#define NVRM_CLASS_VP_G74		0x7476
#define NVRM_CLASS_VP_G98		0x88b2
#define NVRM_CLASS_VP_GT212		0x85b2
#define NVRM_CLASS_VP_GF100		0x90b2
#define NVRM_CLASS_VP_GF119		0x95b2

struct nvrm_create_bsp {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_BSP_G74		0x74b0
#define NVRM_CLASS_BSP_G98		0x88b1
#define NVRM_CLASS_BSP_GT212		0x85b1
#define NVRM_CLASS_BSP_MCP89		0x86b1
#define NVRM_CLASS_BSP_GF100		0x90b1
#define NVRM_CLASS_BSP_GF119		0x95b1
#define NVRM_CLASS_BSP_GM107		0xa0b0 /* hm. */

struct nvrm_create_ppp {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_PPP_G98		0x88b3
#define NVRM_CLASS_PPP_GT212		0x85b3
#define NVRM_CLASS_PPP_GF100		0x90b3

struct nvrm_create_venc {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_VENC_GK104		0x90b7
#define NVRM_CLASS_VENC_GM107		0xc0b7

/* no create param */
#define NVRM_CLASS_VCOMP_MCP89		0x86b6

/* engine 16 - no create param */

#define NVRM_CLASS_UNK95A1		0x95a1

/* evil stuff */

/* no create param */
#define NVRM_CLASS_CRYPT_G74		0x74c1
#define NVRM_CLASS_CRYPT_G98		0x88b4

/* software fifo eng */

struct nvrm_create_sw_unk0075 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
};
#define NVRM_CLASS_SW_UNK0075		0x0075 /* not on G98 nor MCP79 */

/* no create param */
#define NVRM_CLASS_SW_UNK007D		0x007d
#define NVRM_CLASS_SW_UNK208A		0x208a /* not on G98 nor MCP79 */
#define NVRM_CLASS_SW_UNK5080		0x5080
#define NVRM_CLASS_SW_UNK50B0		0x50b0 /* G80:G98 and MCP79 */

struct nvrm_create_sw_unk9072 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
};
#define NVRM_CLASS_SW_UNK9072		0x9072 /* GF100+ */

/* no create param */
#define NVRM_CLASS_SW_UNK9074		0x9074 /* GF100+ */

/* display */

/* no create param */
#define NVRM_CLASS_DISP_ROOT_G80	0x5070 /* wrt device; singleton */
#define NVRM_CLASS_DISP_ROOT_G82	0x8270
#define NVRM_CLASS_DISP_ROOT_G200	0x8370
#define NVRM_CLASS_DISP_ROOT_G94	0x8870
#define NVRM_CLASS_DISP_ROOT_GT212	0x8570
#define NVRM_CLASS_DISP_ROOT_GF119	0x9070
#define NVRM_CLASS_DISP_ROOT_GK104	0x9170
#define NVRM_CLASS_DISP_ROOT_GK110	0x9270
#define NVRM_CLASS_DISP_ROOT_GM107	0x9470

struct nvrm_create_disp_fifo {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_DISP_FIFO		0x5079 /* ... allows you to bind to any of the FIFOs??? */

struct nvrm_create_disp_fifo_pio {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
};
#define NVRM_CLASS_DISP_CURSOR_G80	0x507a
#define NVRM_CLASS_DISP_CURSOR_G82	0x827a
#define NVRM_CLASS_DISP_CURSOR_GT212	0x857a
#define NVRM_CLASS_DISP_CURSOR_GF119	0x907a
#define NVRM_CLASS_DISP_CURSOR_GK104	0x917a

#define NVRM_CLASS_DISP_OVPOS_G80	0x507b
#define NVRM_CLASS_DISP_OVPOS_G82	0x827b
#define NVRM_CLASS_DISP_OVPOS_GT212	0x857b
#define NVRM_CLASS_DISP_OVPOS_GF119	0x907b
#define NVRM_CLASS_DISP_OVPOS_GK104	0x917b

struct nvrm_create_disp_fifo_dma {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
	uint32_t unk18;
	uint32_t unk1c;
};
#define NVRM_CLASS_DISP_FLIP_G80	0x507c
#define NVRM_CLASS_DISP_FLIP_G82	0x827c
#define NVRM_CLASS_DISP_FLIP_G200	0x837c
#define NVRM_CLASS_DISP_FLIP_GT212	0x857c
#define NVRM_CLASS_DISP_FLIP_GF119	0x907c
#define NVRM_CLASS_DISP_FLIP_GK104	0x917c
#define NVRM_CLASS_DISP_FLIP_GK110	0x927c

#define NVRM_CLASS_DISP_MASTER_G80	0x507d
#define NVRM_CLASS_DISP_MASTER_G82	0x827d
#define NVRM_CLASS_DISP_MASTER_G200	0x837d
#define NVRM_CLASS_DISP_MASTER_G94	0x887d
#define NVRM_CLASS_DISP_MASTER_GT212	0x857d
#define NVRM_CLASS_DISP_MASTER_GF119	0x907d
#define NVRM_CLASS_DISP_MASTER_GK104	0x917d
#define NVRM_CLASS_DISP_MASTER_GK110	0x927d
#define NVRM_CLASS_DISP_MASTER_GM107	0x947d

#define NVRM_CLASS_DISP_OVERLAY_G80	0x507e
#define NVRM_CLASS_DISP_OVERLAY_G82	0x827e
#define NVRM_CLASS_DISP_OVERLAY_G200	0x837e
#define NVRM_CLASS_DISP_OVERLAY_GT212	0x857e
#define NVRM_CLASS_DISP_OVERLAY_GF119	0x907e
#define NVRM_CLASS_DISP_OVERLAY_GK104	0x917e

/* scan results */

/* no create param */
#define NVRM_CLASS_UNK0001		0x0001
#define NVRM_CLASS_UNK0074		0x0074
#define NVRM_CLASS_UNK007F		0x007f
#define NVRM_CLASS_UNK402C		0x402c /* wrt subdevice */
#define NVRM_CLASS_UNK507F		0x507f
#define NVRM_CLASS_UNK907F		0x907f
#define NVRM_CLASS_UNK50A0		0x50a0
#define NVRM_CLASS_UNK50E0		0x50e0
#define NVRM_CLASS_UNK50E2		0x50e2
#define NVRM_CLASS_UNK824D		0x824d
#define NVRM_CLASS_UNK884D		0x884d
#define NVRM_CLASS_UNK83CC		0x83cc
#define NVRM_CLASS_UNK844C		0x844c
#define NVRM_CLASS_UNK9067		0x9067 /* wrt device */
#define NVRM_CLASS_UNK90DD		0x90dd /* wrt subdevice */
#define NVRM_CLASS_UNK90E0		0x90e0 /* wrt subdevice */
#define NVRM_CLASS_UNK90E1		0x90e1 /* wrt subdevice */
#define NVRM_CLASS_UNK90E2		0x90e2 /* wrt subdevice */
#define NVRM_CLASS_UNK90E3		0x90e3 /* wrt subdevice */
#define NVRM_CLASS_UNK90E4		0x90e4
#define NVRM_CLASS_UNK90E5		0x90e5
#define NVRM_CLASS_UNK90E6		0x90e6 /* wrt subdevice */
#define NVRM_CLASS_UNK90EC		0x90ec /* wrt device */
#define NVRM_CLASS_UNK9171		0x9171
#define NVRM_CLASS_UNK9271		0x9271
#define NVRM_CLASS_UNK9471		0x9471
#define NVRM_CLASS_UNKA080		0xa080
#define NVRM_CLASS_UNKA0B6		0xa0b6
#define NVRM_CLASS_UNKA0E0		0xa0e0
#define NVRM_CLASS_UNKA0E1		0xa0e1

struct nvrm_create_unk0078 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
};
#define NVRM_CLASS_UNK0078		0x0078

struct nvrm_create_unk007e {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
};
#define NVRM_CLASS_UNK007E		0x007e

struct nvrm_create_unk00db {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_UNK00DB		0x00db

struct nvrm_create_unk00f1 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
};
#define NVRM_CLASS_UNK00F1		0x00f1

struct nvrm_create_unk00ff {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
	uint32_t unk18;
	uint32_t unk1c;
};
#define NVRM_CLASS_UNK00FF		0x00ff

struct nvrm_create_unk25a0 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
};
#define NVRM_CLASS_UNK25A0		0x25a0

struct nvrm_create_unk30f1 {
	uint32_t unk00;
};
#define NVRM_CLASS_UNK30F1		0x30f1

struct nvrm_create_unk30f2 {
	uint32_t unk00;
};
#define NVRM_CLASS_UNK30F2		0x30f2

struct nvrm_create_unk83f3 {
	uint32_t unk00;
};
#define NVRM_CLASS_UNK83F3		0x83f3

struct nvrm_create_unk40ca {
	uint32_t unk00;
};
#define NVRM_CLASS_UNK40CA		0x40ca /* wrt device or subdevice */

struct nvrm_create_unk503b {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_UNK503B		0x503b

struct nvrm_create_unk503c {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
};
#define NVRM_CLASS_UNK503C		0x503c /* wrt subdevice */

struct nvrm_create_unk5072 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
};
#define NVRM_CLASS_UNK5072		0x5072

struct nvrm_create_unk8d75 {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_UNK8D75		0x8d75

struct nvrm_create_unk906d {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_UNK906D		0x906d

struct nvrm_create_unk906e {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_UNK906E		0x906e

struct nvrm_create_unk90f1 {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
};
#define NVRM_CLASS_UNK90F1		0x90f1 /* wrt device */

struct nvrm_create_unka06c {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
};
#define NVRM_CLASS_UNKA06C		0xa06c

struct nvrm_create_unka0b7 {
	uint32_t unk00;
	uint32_t unk04;
};
#define NVRM_CLASS_UNKA0B7		0xa0b7

#endif
