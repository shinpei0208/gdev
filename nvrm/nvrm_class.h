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

#ifndef NVRM_CLASS_H
#define NVRM_CLASS_H

/* sw objects */

#define NVRM_CLASS_CONTEXT_NEW		0x0000
#define NVRM_CLASS_CONTEXT		0x0041

#define NVRM_CLASS_DEVICE		0x0080	/* wrt context */
#define NVRM_CLASS_SUBDEVICE		0x2080	/* wrt device */

#define NVRM_CLASS_TIMER		0x0004	/* wrt subdevice */

#define NVRM_CLASS_VSPACE		0x0070	/* wrt device */

#define NVRM_CLASS_UNK0071		0x0071	/* wrt device */

#define NVRM_CLASS_UNK003E		0x003e	/* wrt device */

struct nvrm_create_unk0073 {
	/* XXX */
};
#define NVRM_CLASS_UNK0073		0x0073	/* wrt device */

struct nvrm_create_unk208f {
	/* XXX */
};
#define NVRM_CLASS_UNK208F		0x208f	/* wrt subdevice */

struct nvrm_create_event {
	/* XXX */
};
#define NVRM_CLASS_EVENT		0x0079	/* wrt graph */

#define NVRM_CLASS_MEMORY		0x0002	/* wrt vspace */

struct nvrm_create_unk83de {
	/* XXX */
};
#define NVRM_CLASS_UNK83DE		0x83de	/* wrt context */

#define NVRM_CLASS_UNK85B6		0x85b6	/* wrt subdevice */

#define NVRM_CLASS_UNK90E0		0x90e0	/* wrt subdevice */
#define NVRM_CLASS_UNK90E1		0x90e1	/* wrt subdevice */

/* FIFO */

struct nvrm_create_fifo_ib {
	uint32_t error_notify;
	uint32_t dma;
	uint64_t ib_addr;
	uint64_t ib_entries;
};
#define NVRM_CLASS_FIFO_IB_G80		0x506f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_G82		0x826f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_GF100	0x906f	/* wrt device */
#define NVRM_CLASS_FIFO_IB_GK104	0xa06f	/* wrt device */

/* graph */

#define NVRM_CLASS_GR_NULL		0x0030

#define NVRM_CLASS_GR_BETA1_NV1		0x0012
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
#define NVRM_CLASS_GR_LIN_NV30		0x305d
#define NVRM_CLASS_GR_SIFC_NV30		0x3066
#define NVRM_CLASS_GR_TEX_NV30		0x307b
#define NVRM_CLASS_GR_SURF2D_G8		0x5062
#define NVRM_CLASS_GR_SIFM_G80		0x5089

#define NVRM_CLASS_GR_2D_G80		0x502d
#define NVRM_CLASS_GR_M2MF_G80		0x5039
#define NVRM_CLASS_GR_3D_G80		0x5097
#define NVRM_CLASS_GR_COMPUTE_G80	0x50c0

#define NVRM_CLASS_GR_3D_G82		0x8297
#define NVRM_CLASS_GR_3D_G200		0x8397
#define NVRM_CLASS_GR_3D_GT212		0x8597
#define NVRM_CLASS_GR_COMPUTE_GT212	0x85c0

#define NVRM_CLASS_GR_2D_GF100		0x902d
#define NVRM_CLASS_GR_M2MF_GF100	0x9039
#define NVRM_CLASS_GR_3D_GF100		0x9097
#define NVRM_CLASS_GR_COMPUTE_GF100	0x90c0

#define NVRM_CLASS_GR_COMPUTE_GF110	0x91c0
#define NVRM_CLASS_GR_3D_GF108		0x9197
#define NVRM_CLASS_GR_3D_GF110		0x9297

#define NVRM_CLASS_GR_UPLOAD_GK104	0xa040
#define NVRM_CLASS_GR_ENG3D_GK104	0xa097
#define NVRM_CLASS_GR_COMPUTE_GK104	0xa0c0

/* copy */

struct nvrm_create_copy {
	/* XXX */
};
#define NVRM_CLASS_COPY_GT212		0x85b5
#define NVRM_CLASS_COPY0_GF100		0x90b5
#define NVRM_CLASS_COPY1_GF100		0x90b8 /* XXX: wtf? */
#define NVRM_CLASS_COPY_GK104		0xa0b5

/* vdec etc. */

#define NVRM_CLASS_MPEG_NV31		0x3174
#define NVRM_CLASS_MPEG_G82		0x8274

#define NVRM_CLASS_ME_NV40		0x4075

#define NVRM_CLASS_VP_G80		0x5076
#define NVRM_CLASS_VP_G74		0x7476
#define NVRM_CLASS_VP_G98		0x88b2
#define NVRM_CLASS_VP_GT212		0x85b2
#define NVRM_CLASS_VP_GF100		0x90b2
#define NVRM_CLASS_VP_GF119		0x95b2

#define NVRM_CLASS_BSP_G74		0x74b0
#define NVRM_CLASS_BSP_G98		0x88b1
#define NVRM_CLASS_BSP_GT212		0x85b1
#define NVRM_CLASS_BSP_GF100		0x90b1
#define NVRM_CLASS_BSP_GF119		0x95b1

#define NVRM_CLASS_PPP_G98		0x88b3
#define NVRM_CLASS_PPP_GT212		0x85b3
#define NVRM_CLASS_PPP_GF100		0x90b3

#define NVRM_CLASS_VENC_GK104		0x90b7

/* evil stuff */

struct nvrm_create_crypt {
	/* XXX */
};
#define NVRM_CLASS_CRYPT_G74		0x74c1
#define NVRM_CLASS_CRYPT_G98		0x88b4

/* software fifo eng */

#define NVRM_CLASS_SW_UNK0075		0x0075 /* not on G98 nor MCP79 */
#define NVRM_CLASS_SW_UNK007D		0x007d
#define NVRM_CLASS_SW_UNK208A		0x208a /* not on G98 nor MCP79 */
#define NVRM_CLASS_SW_UNK5080		0x5080
#define NVRM_CLASS_SW_UNK50B0		0x50b0 /* G80:G98 and MCP79 */
#define NVRM_CLASS_SW_UNK9072		0x9072 /* GF100+ */
#define NVRM_CLASS_SW_UNK9074		0x9074 /* GF100+ */

#endif
