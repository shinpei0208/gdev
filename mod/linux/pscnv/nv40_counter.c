/*
 * Copyright (C) - Nouveau Community
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: Martin Peres
 */

#include "drmP.h"
#include "nouveau_drv.h"
#include "nouveau_pm.h"

static void
pcounter_counters_readout_periodic(unsigned long data);

int
nv40_counter_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;

	/* initialise the periodic timer */
	setup_timer(&counter->readout_timer,
		    pcounter_counters_readout_periodic, (unsigned long)dev);

	return 0;
}

void
nv40_counter_fini(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	int set, sig;

	nv40_counter_stop(dev);

	for (set = 0; set < 8; set++)
		for (sig = 0; sig < 4; sig++)
			counter->sets[set].signals[sig] = 0;
}

static u8
nv40_counter_signal(struct drm_device *dev, enum nouveau_counter_signal s,
		    u8 *set, u8 *signal)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	u8 chipset = dev_priv->chipset;

	if (set)
		*set = 0;
	if (signal)
		*signal = 0;

	switch (s) {
		case NONE:
		{
			return 0;
		}
		case PGRAPH_IDLE:
		{
			if (set)
				*set = 1;
			if (!signal)
				return 0;

			switch (chipset)
			{
				case 0x50:
					*signal = 0xc8;
					break;
				case 0x84:
				case 0x86:
				case 0x98:
					*signal = 0xbd;
					break;
				case 0xa0:
				case 0xac:
					*signal = 0xc9;
					break;
				case 0xa3:
				case 0xa5:
				case 0xa8:
					*signal = 0xcb;
					break;
				default:
					return -ENOENT;
			}

			return 0;
		}
		case PGRAPH_INTR_PENDING:
		{
			if (set)
				*set = 1;
			if (!signal)
				return 0;

			switch (chipset)
			{
				case 0x50:
					*signal = 0xca;
					break;
				case 0x84:
				case 0x86:
				case 0x98:
					*signal = 0xbf;
					break;
				case 0xa0:
				case 0xac:
					*signal = 0xcb;
					break;
				case 0xa3:
				case 0xa5:
				case 0xa8:
					*signal = 0xcd;
					break;
				default:
					return -ENOENT;
			}

			return 0;
		}
		case CTXPROG_ACTIVE:
		{
			if (set)
				*set = 1;
			if (!signal)
				return 0;

			switch (chipset)
			{
				case 0x50:
					*signal = 0xd2;
					break;
				case 0x84:
				case 0x86:
				case 0x98:
					*signal = 0xc7;
					break;
				case 0xa0:
				case 0xac:
					*signal = 0x1c;
					break;
				case 0xa3:
				case 0xa5:
				case 0xa8:
					*signal = 0xd5;
					break;
				default:
					return -ENOENT;
			}

			return 0;
		}
	};

	return -ENOENT;
}

static void
nv40_counter_reprogram(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;
	int set;

	spin_lock_irqsave(&counter->counter_lock, flags);

	for (set = 0; set < 8; set++) {
		nv_wr32(dev, 0xa7c0 + set * 4, 0x1);
		nv_wr32(dev, 0xa500 + set * 4, 0);
		nv_wr32(dev, 0xa520 + set * 4, 0);

		nv_wr32(dev, 0xa400 + set * 4, counter->signals[set][0]);
		nv_wr32(dev, 0xa440 + set * 4, counter->signals[set][1]);
		nv_wr32(dev, 0xa480 + set * 4, counter->signals[set][2]);
		nv_wr32(dev, 0xa4c0 + set * 4, counter->signals[set][3]);

		nv_wr32(dev, 0xa420 + set * 4, 0xaaaa);
		nv_wr32(dev, 0xa460 + set * 4, 0xaaaa);
		nv_wr32(dev, 0xa4a0 + set * 4, 0xaaaa);
		nv_wr32(dev, 0xa4e0 + set * 4, 0xaaaa);
	}

	/* reset the counters */
	nv_mask(dev, 0x400084, 0x20, 0x20);

	spin_unlock_irqrestore(&counter->counter_lock, flags);
}

static void
pcounter_counters_readout(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;
	int s;

	spin_lock_irqsave(&counter->counter_lock, flags);

	/* readout */
	nv_mask(dev, 0x400084, 0x0, 0x20);

	for (s = 0; s < 8; s++) {
		counter->sets[s].cycles = nv_rd32(dev, 0xa600 + s * 4);
		counter->sets[s].signals[0] = nv_rd32(dev, 0xa700 + s * 4);
		counter->sets[s].signals[1] = nv_rd32(dev, 0xa6c0 + s * 4);
		counter->sets[s].signals[2] = nv_rd32(dev, 0xa680 + s * 4);
		counter->sets[s].signals[3] = nv_rd32(dev, 0xa740 + s * 4);
	}

	if (counter->periodic_polling)
		mod_timer(&counter->readout_timer, jiffies + (HZ / 10));

	spin_unlock_irqrestore(&counter->counter_lock, flags);

	if (counter->on_update)
		counter->on_update(dev);
}

int nv40_counter_watch_signal(struct drm_device *dev,
			    enum nouveau_counter_signal wanted_signal)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;
	u8 signal, set;
	int i, ret;

	ret = nv40_counter_signal(dev, wanted_signal, &set, &signal);
	if (ret)
		return ret;

	spin_lock_irqsave(&counter->counter_lock, flags);

	for (i = 0; i < 4; i++) {
		if (counter->signals[set][i] == 0 ||
		    counter->signals[set][i] == signal) {
			counter->signals[set][i] = signal;
			spin_unlock_irqrestore(&counter->counter_lock, flags);
			return 0;
		}
	}

	spin_unlock_irqrestore(&counter->counter_lock, flags);

	return -ENOSPC;
}

int nv40_counter_unwatch_signal(struct drm_device *dev,
			    enum nouveau_counter_signal wanted_signal)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;
	u8 signal, set;
	int i, ret;

	ret = nv40_counter_signal(dev, wanted_signal, &set, &signal);
	if (ret)
		return ret;

	spin_lock_irqsave(&counter->counter_lock, flags);

	for (i = 0; i < 4; i++) {
		if (counter->signals[set][i] == signal) {
			counter->signals[set][i] = 0;
			spin_unlock_irqrestore(&counter->counter_lock, flags);
			return 0;
		}
	}

	spin_unlock_irqrestore(&counter->counter_lock, flags);

	return -ENOENT;
}

void
nv40_counter_poll(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;
	bool exit = 0;

	/* do not poll if continuous polling is done */
	spin_lock_irqsave(&counter->counter_lock, flags);
	if (counter->periodic_polling)
		exit = 1;
	spin_unlock_irqrestore(&counter->counter_lock, flags);
	if (exit)
		return;

	nv40_counter_reprogram(dev);
	msleep(100);
	pcounter_counters_readout(dev);
}

static void
pcounter_counters_readout_periodic(unsigned long data)
{
	struct drm_device *dev = (struct drm_device *)data;
	pcounter_counters_readout(dev);
}

void
nv40_counter_start(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;

	nv40_counter_reprogram(dev);

	/* does it need to be atomic? */
	spin_lock_irqsave(&counter->counter_lock, flags);

	counter->periodic_polling = 1;
	mod_timer(&counter->readout_timer, jiffies + (HZ / 10));

	spin_unlock_irqrestore(&counter->counter_lock, flags);
}

void
nv40_counter_stop(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;

	spin_lock_irqsave(&counter->counter_lock, flags);

	counter->periodic_polling = 0;

	del_timer_sync(&counter->readout_timer);
	spin_unlock_irqrestore(&counter->counter_lock, flags);
}

int
nv40_counter_value(struct drm_device *dev, enum nouveau_counter_signal signal,
		   u32 *val, u32 *count)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_pm_counter *counter = &dev_priv->engine.pm.counter;
	unsigned long flags;
	u8 set, sig, i;

	nv40_counter_signal(dev, signal, &set, &sig);

	spin_lock_irqsave(&counter->counter_lock, flags);
	for (i = 0; i < 4; i++) {
		if (counter->signals[set][i] == sig) {
			*count = counter->sets[set].cycles;
			*val = counter->sets[set].signals[i];
			spin_unlock_irqrestore(&counter->counter_lock, flags);
			return 0;
		}
	}
	spin_unlock_irqrestore(&counter->counter_lock, flags);

	return -ENOENT;
}
