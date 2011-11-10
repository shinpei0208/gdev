/*
 * Copyright (C) The Weather Channel, Inc.  2002.  All Rights Reserved.
 * Copyright 2005 Stephane Marchesin
 *
 * The Weather Channel (TM) funded Tungsten Graphics to develop the
 * initial release of the Radeon 8500 driver under the XFree86 license.
 * This notice must be preserved.
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
 * THE AUTHORS AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Keith Whitwell <keith@tungstengraphics.com>
 */


#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"

void
nouveau_mem_timing_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_engine *pm = &dev_priv->engine.pm;
	struct nouveau_pm_memtimings *memtimings = &pm->memtimings;
	struct nvbios *bios = &dev_priv->vbios;
	struct bit_entry P;
	u8 tUNK_0, tUNK_1, tUNK_2;
	u8 tRP;		/* Byte 3 */
	u8 tRAS;	/* Byte 5 */
	u8 tRFC;	/* Byte 7 */
	u8 tRC;		/* Byte 9 */
	u8 tUNK_10, tUNK_11, tUNK_12, tUNK_13, tUNK_14;
	u8 tUNK_18, tUNK_19, tUNK_20, tUNK_21;
	u8 *mem = NULL, *entry;
	int i, recordlen, entries;

	if (bios->type == NVBIOS_BIT) {
		if (bit_table(dev, 'P', &P))
			return;

		if (P.version == 1)
			mem = ROMPTR(bios, P.data[4]);
		else
		if (P.version == 2)
			mem = ROMPTR(bios, P.data[8]);
		else {
			NV_WARN(dev, "unknown mem for BIT P %d\n", P.version);
		}
	} else {
		NV_DEBUG(dev, "BMP version too old for memory\n");
		return;
	}

	if (!mem) {
		NV_DEBUG(dev, "memory timing table pointer invalid\n");
		return;
	}

	if (mem[0] != 0x10) {
		NV_WARN(dev, "memory timing table 0x%02x unknown\n", mem[0]);
		return;
	}

	/* validate record length */
	entries   = mem[2];
	recordlen = mem[3];
	if (recordlen < 15) {
		NV_ERROR(dev, "mem timing table length unknown: %d\n", mem[3]);
		return;
	}

	/* parse vbios entries into common format */
	memtimings->timing =
		kcalloc(entries, sizeof(*memtimings->timing), GFP_KERNEL);
	if (!memtimings->timing)
		return;

	entry = mem + mem[1];
	for (i = 0; i < entries; i++, entry += recordlen) {
		struct nouveau_pm_memtiming *timing = &pm->memtimings.timing[i];
		if (entry[0] == 0)
			continue;

		tUNK_18 = 1;
		tUNK_19 = 1;
		tUNK_20 = 0;
		tUNK_21 = 0;
		switch (recordlen>22?22:recordlen) {
		case 0x22:
			tUNK_21 = entry[21];
		case 0x21:
			tUNK_20 = entry[20];
		case 0x20:
			tUNK_19 = entry[19];
		case 0x19:
			tUNK_18 = entry[18];
		default:
			tUNK_0  = entry[0];
			tUNK_1  = entry[1];
			tUNK_2  = entry[2];
			tRP     = entry[3];
			tRAS    = entry[5];
			tRFC    = entry[7];
			tRC     = entry[9];
			tUNK_10 = entry[10];
			tUNK_11 = entry[11];
			tUNK_12 = entry[12];
			tUNK_13 = entry[13];
			tUNK_14 = entry[14];
			break;
		}

		timing->reg_100220 = (tRC << 24 | tRFC << 16 | tRAS << 8 | tRP);

		/* XXX: I don't trust the -1's and +1's... they must come
		 *      from somewhere! */
		timing->reg_100224 = ((tUNK_0 + tUNK_19 + 1) << 24 |
				      tUNK_18 << 16 |
				      (tUNK_1 + tUNK_19 + 1) << 8 |
				      (tUNK_2 - 1));

		timing->reg_100228 = (tUNK_12 << 16 | tUNK_11 << 8 | tUNK_10);
		if(recordlen > 19) {
			timing->reg_100228 += (tUNK_19 - 1) << 24;
		} else {
			timing->reg_100228 += tUNK_12 << 24;
		}

		/* XXX: reg_10022c */

		timing->reg_100230 = (tUNK_20 << 24 | tUNK_21 << 16 |
				      tUNK_13 << 8  | tUNK_13);

		/* XXX: +6? */
		timing->reg_100234 = (tRAS << 24 | (tUNK_19 + 6) << 8 | tRC);
		if(tUNK_10 > tUNK_11) {
			timing->reg_100234 += tUNK_10 << 16;
		} else {
			timing->reg_100234 += tUNK_11 << 16;
		}

		/* XXX; reg_100238, reg_10023c */
		NV_DEBUG(dev, "Entry %d: 220: %08x %08x %08x %08x\n", i,
			 timing->reg_100220, timing->reg_100224,
			 timing->reg_100228, timing->reg_10022c);
		NV_DEBUG(dev, "         230: %08x %08x %08x %08x\n",
			 timing->reg_100230, timing->reg_100234,
			 timing->reg_100238, timing->reg_10023c);

		NV_DEBUG(dev, "         tUNK_14 %08x\n", tUNK_14);
	}

	memtimings->nr_timing  = entries;
	memtimings->supported = true;
}

void
nouveau_mem_timing_fini(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_memtimings *mem = &dev_priv->engine.pm.memtimings;

	kfree(mem->timing);
}
