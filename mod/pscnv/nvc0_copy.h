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

#ifndef __NVC0_COPY_H__
#define __NVC0_COPY_H__

#define NVC0_COPY(x) container_of(x, struct nvc0_copy_engine, base)

struct nvc0_copy_engine {
	struct pscnv_engine base;
	spinlock_t lock;
	uint32_t irq;
	uint32_t pmc;
	uint32_t fuc;
	uint32_t ctx;
	int id;
};

#endif
