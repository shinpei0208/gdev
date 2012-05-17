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

#ifndef __NVC0_GRAPH_H__
#define __NVC0_GRAPH_H__

#define NVC0_TP_MAX 32
#define NVC0_GPC_MAX 4

#define NVC0_GRAPH(x) container_of(x, struct nvc0_graph_engine, base)

struct nvc0_graph_engine {
	struct pscnv_engine base;
	uint32_t grctx_size;
	uint32_t *grctx_initvals;
	uint8_t ropc_count;
	uint8_t gpc_count;
	uint8_t tp_count;
	uint8_t gpc_tp_count[NVC0_GPC_MAX];
	uint8_t gpc_cx_count[NVC0_GPC_MAX];
	struct pscnv_bo *obj188b4;
	struct pscnv_bo *obj188b8;
	struct pscnv_bo *obj08004;
	struct pscnv_bo *obj0800c;
	struct pscnv_bo *obj19848;
	uint32_t magic_val; /* XXX */
};

/* nvc0_graph.c uses this also to determine supported chipsets */
static inline u32
nvc0_graph_class(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;

	switch (dev_priv->chipset) {
	case 0xc0:
	case 0xc3:
	case 0xc4:
	case 0xce: /* guess, mmio trace shows only 0x9097 state */
	case 0xcf: /* guess, mmio trace shows only 0x9097 state */
		return 0x9097;
	case 0xc1:
		return 0x9197;
	case 0xc8:
		return 0x9297;
	default:
		return 0;
	}
}

extern int nvc0_grctx_construct(struct drm_device *dev,
								struct nvc0_graph_engine *graph,
								struct pscnv_chan *chan);

#endif
