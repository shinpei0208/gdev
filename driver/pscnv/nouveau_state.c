/*
 * Copyright 2005 Stephane Marchesin
 * Copyright 2008 Stuart Bennett
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
 * PRECISION INSIGHT AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <linux/swab.h>
#include <linux/slab.h>
#include "drmP.h"
#include "drm.h"
#include "drm_sarea.h"
#include "drm_crtc_helper.h"
#include <linux/vgaarb.h>
#include <linux/vga_switcheroo.h>

#include "nouveau_drv.h"
#include "pscnv_drm.h"
#include "nouveau_reg.h"
#include "nouveau_fbcon.h"
#include "nouveau_pm.h"
#include "nv50_display.h"
#include "pscnv_vm.h"
#include "pscnv_chan.h"
#include "pscnv_fifo.h"
#include "pscnv_ioctl.h"

static void nouveau_stub_takedown(struct drm_device *dev) {}
static int nouveau_stub_init(struct drm_device *dev) { return 0; }

static int nouveau_init_engine_ptrs(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_engine *engine = &dev_priv->engine;

	if (dev_priv->chipset < 0x10) {
		engine->gpio.init		= nouveau_stub_init;
		engine->gpio.takedown		= nouveau_stub_takedown;
		engine->gpio.get		= NULL;
		engine->gpio.set		= NULL;
		engine->gpio.irq_enable		= NULL;
		engine->pm.clocks_get		= nv04_pm_clocks_get;
		engine->pm.clocks_pre		= nv04_pm_clocks_pre;
		engine->pm.clocks_set		= nv04_pm_clocks_set;
	} else if (dev_priv->chipset < 0x50 || (dev_priv->chipset & 0xf0) == 0x60) {
		engine->gpio.init		= nouveau_stub_init;
		engine->gpio.takedown		= nouveau_stub_takedown;
		engine->gpio.get		= nv10_gpio_get;
		engine->gpio.set		= nv10_gpio_set;
		engine->gpio.irq_enable		= NULL;
		engine->pm.clocks_get		= nv04_pm_clocks_get;
		engine->pm.clocks_pre		= nv04_pm_clocks_pre;
		engine->pm.clocks_set		= nv04_pm_clocks_set;
	} else {
		engine->gpio.init		= nv50_gpio_init;
		engine->gpio.takedown		= nouveau_stub_takedown;
		engine->gpio.get		= nv50_gpio_get;
		engine->gpio.set		= nv50_gpio_set;
		engine->gpio.irq_enable		= nv50_gpio_irq_enable;
		engine->pm.pwm_get		= nv50_pm_pwm_get;
		engine->pm.pwm_set		= nv50_pm_pwm_set;
		engine->pm.counter.init		= nv40_counter_init;
		engine->pm.counter.takedown	= nv40_counter_fini;
		engine->pm.counter.watch	= nv40_counter_watch_signal;
		engine->pm.counter.unwatch	= nv40_counter_unwatch_signal;
		engine->pm.counter.poll		= nv40_counter_poll;
		engine->pm.counter.start	= nv40_counter_start;
		engine->pm.counter.stop		= nv40_counter_stop;
		engine->pm.counter.signal_value	= nv40_counter_value;
		switch (dev_priv->chipset) {
		case 0xa3:
		case 0xa5:
		case 0xa8:
		case 0xaf:
			engine->pm.clocks_get	= nva3_pm_clocks_get;
			engine->pm.clocks_pre	= nva3_pm_clocks_pre;
			engine->pm.clocks_set	= nva3_pm_clocks_set;
			break;
		case 0xc0:
			engine->pm.clocks_get		= nvc0_pm_clocks_get;
			engine->pm.clocks_pre		= nvc0_pm_clocks_pre;
			engine->pm.clocks_set		= nvc0_pm_clocks_set;
			break;
		default:
			engine->pm.clocks_get	= nv50_pm_clocks_get;
			engine->pm.clocks_pre	= nv50_pm_clocks_pre;
			engine->pm.clocks_set	= nv50_pm_clocks_set;
			break;
		}
	}

	if (dev_priv->chipset < 0x40) {
	} else if (dev_priv->chipset < 0x80) {
		engine->pm.temp_get		= nv40_temp_get;
	} else {
		engine->pm.temp_get		= nv84_temp_get;
	}

	if (dev_priv->chipset >= 0x30 && dev_priv->chipset < 0xc0) {
		engine->pm.voltage_get		= nouveau_voltage_gpio_get;
		engine->pm.voltage_set		= nouveau_voltage_gpio_set;
	}

	if (dev_priv->chipset < 0x50 || (dev_priv->chipset & 0xf0) == 0x60) {
#if 0
		engine->display.early_init	= nv04_display_early_init;
		engine->display.late_takedown	= nv04_display_late_takedown;
		engine->display.create		= nv04_display_create;
		engine->display.init		= nv04_display_init;
		engine->display.destroy		= nv04_display_destroy;
#endif
		NV_ERROR(dev, "NV%02x unsupported\n", dev_priv->chipset);
		return -ENOSYS;
	} else {
		engine->display.early_init	= nv50_display_early_init;
		engine->display.late_takedown	= nv50_display_late_takedown;
		engine->display.create		= nv50_display_create;
		engine->display.init		= nv50_display_init;
		engine->display.destroy		= nv50_display_destroy;
	}

	return 0;
}

static unsigned int
nouveau_vga_set_decode(void *priv, bool state)
{
	struct drm_device *dev = priv;
	struct drm_nouveau_private *dev_priv = dev->dev_private;

	if (dev_priv->chipset >= 0x40)
		nv_wr32(dev, 0x88054, state);
	else
		nv_wr32(dev, 0x1854, state);

	if (state)
		return VGA_RSRC_LEGACY_IO | VGA_RSRC_LEGACY_MEM |
		       VGA_RSRC_NORMAL_IO | VGA_RSRC_NORMAL_MEM;
	else
		return VGA_RSRC_NORMAL_IO | VGA_RSRC_NORMAL_MEM;
}

static void nouveau_switcheroo_set_state(struct pci_dev *pdev,
					 enum vga_switcheroo_state state)
{
	pm_message_t pmm = { .event = PM_EVENT_SUSPEND };
	if (state == VGA_SWITCHEROO_ON) {
		printk(KERN_ERR "VGA switcheroo: switched nouveau on\n");
		nouveau_pci_resume(pdev);
	} else {
		printk(KERN_ERR "VGA switcheroo: switched nouveau off\n");
		nouveau_pci_suspend(pdev, pmm);
	}
}

static void nouveau_switcheroo_reprobe(struct pci_dev *pdev)
{
	struct drm_device *dev = pci_get_drvdata(pdev);
	nouveau_fbcon_output_poll_changed(dev);
}

static bool nouveau_switcheroo_can_switch(struct pci_dev *pdev)
{
	struct drm_device *dev = pci_get_drvdata(pdev);
	bool can_switch;

	spin_lock(&dev->count_lock);
	can_switch = (dev->open_count == 0);
	spin_unlock(&dev->count_lock);
	return can_switch;
}

int
nouveau_card_init(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	struct nouveau_engine *engine;
	int ret;
	int i;

	NV_DEBUG(dev, "prev state = %d\n", dev_priv->init_state);

	if (dev_priv->init_state == NOUVEAU_CARD_INIT_DONE)
		return 0;

	NV_INFO(dev, "Initializing card...\n");

	vga_client_register(dev->pdev, dev, NULL, nouveau_vga_set_decode);
#ifdef PSCNV_KAPI_SWITCHEROO_REPROBE
	vga_switcheroo_register_client(dev->pdev, nouveau_switcheroo_set_state,
								   nouveau_switcheroo_can_switch);
#else
	vga_switcheroo_register_client(dev->pdev, nouveau_switcheroo_set_state,
								   nouveau_switcheroo_reprobe,
								   nouveau_switcheroo_can_switch);
#endif

	dev_priv->init_state = NOUVEAU_CARD_INIT_FAILED;

	/* Initialise internal driver API hooks */
	ret = nouveau_init_engine_ptrs(dev);
	if (ret)
		goto out;
	engine = &dev_priv->engine;
	spin_lock_init(&dev_priv->context_switch_lock);

	/* Make the CRTCs and I2C buses accessible */
	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		ret = engine->display.early_init(dev);
		if (ret)
			goto out;
	}

	/* Parse BIOS tables / Run init tables if card not POSTed */
	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		ret = nouveau_bios_init(dev);
		if (ret)
			goto out_display_early;

		/* workaround an odd issue on nvc1 by disabling the device's
		 * nosnoop capability.  hopefully won't cause issues until a
		 * better fix is found - assuming there is one...
		 */
		if (dev_priv->chipset == 0xc1) {
			nv_mask(dev, 0x00088080, 0x00000800, 0x00000000);
		}
	}

	ret = pscnv_mem_init(dev);
	if (ret)
		goto out_bios;

	nouveau_pm_init(dev);

	switch (dev_priv->card_type) {
		case NV_50:
			ret = nv50_chan_init(dev);
			break;
		case NV_C0:
			ret = nvc0_chan_init(dev);
			break;
		default:
			NV_ERROR(dev, "No CHAN implementation for NV%02x!\n", dev_priv->chipset);
			ret = -ENOSYS;
	}
	if (ret)
		goto out_vram;

	switch (dev_priv->card_type) {
		case NV_50:
			ret = nv50_vm_init(dev);
			break;
		case NV_C0:
			ret = nvc0_vm_init(dev);
			break;
		default:
			NV_ERROR(dev, "No VM implementation for NV%02x!\n", dev_priv->chipset);
			ret = -ENOSYS;
	}
	if (ret)
		goto out_chan;

	/* PMC */
	nv_wr32(dev, NV03_PMC_ENABLE, 0xFFFFFFFF);

	/* PBUS */
	nv_wr32(dev, 0x1100, 0xFFFFFFFF);
	nv_wr32(dev, 0x1140, 0xFFFFFFFF);

	/* PGPIO */
	ret = engine->gpio.init(dev);
	if (ret)
		goto out_vm;

	/* PTIMER */
	ret = nv04_timer_init(dev);
	if (ret)
		goto out_gpio;

	/* XXX: handle noaccel */
	switch (dev_priv->card_type) {
		case NV_50:
			/* PFIFO */
			ret = nv50_fifo_init(dev);
			if (!ret) {
				/* PGRAPH */
				nv50_graph_init(dev);
			}
			break;
		case NV_C0:
			/* PFIFO */
			ret = nvc0_fifo_init(dev);
			if (!ret) {
				/* PGRAPH */
				ret = nvc0_graph_init(dev);
				if (!ret) {
					/* PCOPY0 */
					nvc0_copy_init(dev, 0);
					/* PCOPY1 */
					nvc0_copy_init(dev, 1);
				}
			}
			break;
		default:
			break;
	}
	switch (dev_priv->chipset) {
		case 0x84:
		case 0x86:
		case 0x92:
		case 0x94:
		case 0x96:
		case 0xa0:
			nv84_crypt_init(dev);
			break;
		case 0x98:
		case 0xaa:
		case 0xac:
			nv98_crypt_init(dev);
			break;
	}

	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		ret = engine->display.create(dev);
		if (ret)
			goto out_fifo;
	}

	/* this call irq_preinstall, register irq handler and
	 * call irq_postinstall
	 */
	ret = drm_irq_install(dev);
	if (ret)
		goto out_display;

	ret = drm_vblank_init(dev, 0);
	if (ret)
		goto out_irq;

	/* what about PVIDEO/PCRTC/PRAMDAC etc? */
#if 0
	if (!engine->graph.accel_blocked) {
		ret = nouveau_card_init_channel(dev);
		if (ret)
			goto out_irq;
	}
#endif

	ret = nouveau_backlight_init(dev);
	if (ret)
		NV_ERROR(dev, "Error %d registering backlight\n", ret);

	dev_priv->init_state = NOUVEAU_CARD_INIT_DONE;

	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		nouveau_fbcon_init(dev);
		drm_kms_helper_poll_init(dev);
	}

	NV_INFO(dev, "Card initialized.\n");
	return 0;

#if 0
out_channel:
	if (dev_priv->channel) {
		nouveau_channel_free(dev_priv->channel);
		dev_priv->channel = NULL;
	}
#endif
out_irq:
	drm_irq_uninstall(dev);
out_display:
	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		engine->display.destroy(dev);
	}
out_fifo:
	for (i = 0; i < PSCNV_ENGINES_NUM; i++)
		if (dev_priv->engines[i]) {
			dev_priv->engines[i]->takedown(dev_priv->engines[i]);
			dev_priv->engines[i] = 0;
		}
	if (dev_priv->fifo)
		dev_priv->fifo->takedown(dev);
out_gpio:
	engine->gpio.takedown(dev);
out_vm:
	nv_wr32(dev, 0x1140, 0);
	dev_priv->vm->takedown(dev);
out_chan:
	dev_priv->chan->takedown(dev);
out_vram:
	pscnv_mem_takedown(dev);
out_bios:
	nouveau_pm_fini(dev);
	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		nouveau_bios_takedown(dev);
	}
out_display_early:
	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		engine->display.late_takedown(dev);
	}
out:
	vga_client_register(dev->pdev, NULL, NULL, NULL);
	return ret;
}

static void nouveau_card_takedown(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	int i;

	NV_DEBUG(dev, "prev state = %d\n", dev_priv->init_state);

	if (dev_priv->init_state != NOUVEAU_CARD_INIT_DOWN) {
		NV_INFO(dev, "Stopping card...\n");
		nouveau_backlight_exit(dev);
		drm_irq_uninstall(dev);
		for (i = 0; i < PSCNV_ENGINES_NUM; i++)
			if (dev_priv->engines[i]) {
				dev_priv->engines[i]->takedown(dev_priv->engines[i]);
				dev_priv->engines[i] = 0;
			}
		if (dev_priv->fifo)
			dev_priv->fifo->takedown(dev);
		dev_priv->vm->takedown(dev);
		dev_priv->chan->takedown(dev);
		pscnv_mem_takedown(dev);
		nv_wr32(dev, 0x1140, 0);
		nouveau_pm_fini(dev);
		nouveau_bios_takedown(dev);

		vga_client_register(dev->pdev, NULL, NULL, NULL);

		dev_priv->init_state = NOUVEAU_CARD_INIT_DOWN;
		NV_INFO(dev, "Card stopped.\n");
	}
}

/* here a client dies, release the stuff that was allocated for its
 * file_priv */
void nouveau_preclose(struct drm_device *dev, struct drm_file *file_priv)
{
	pscnv_chan_cleanup(dev, file_priv);
	pscnv_vspace_cleanup(dev, file_priv);
}

/* first module load, setup the mmio/fb mapping */
/* KMS: we need mmio at load time, not when the first drm client opens. */
int nouveau_firstopen(struct drm_device *dev)
{
	nouveau_card_init(dev);
	return 0;
}

/* if we have an OF card, copy vbios to RAMIN */
static void nouveau_OF_copy_vbios_to_ramin(struct drm_device *dev)
{
#if defined(__powerpc__)
	int size, i;
	const uint32_t *bios;
	struct device_node *dn = pci_device_to_OF_node(dev->pdev);
	if (!dn) {
		NV_INFO(dev, "Unable to get the OF node\n");
		return;
	}

	bios = of_get_property(dn, "NVDA,BMP", &size);
	if (bios) {
		for (i = 0; i < size; i += 4)
			nv_wi32(dev, i, bios[i/4]);
		NV_INFO(dev, "OF bios successfully copied (%d bytes)\n", size);
	} else {
		NV_INFO(dev, "Unable to get the OF bios\n");
	}
#endif
}

static struct apertures_struct *nouveau_get_apertures(struct drm_device *dev)
{
	struct pci_dev *pdev = dev->pdev;
	struct apertures_struct *aper = alloc_apertures(3);
	if (!aper)
		return NULL;

	aper->ranges[0].base = pci_resource_start(pdev, 1);
	aper->ranges[0].size = pci_resource_len(pdev, 1);
	aper->count = 1;

	if (pci_resource_len(pdev, 2)) {
		aper->ranges[aper->count].base = pci_resource_start(pdev, 2);
		aper->ranges[aper->count].size = pci_resource_len(pdev, 2);
		aper->count++;
	}

	if (pci_resource_len(pdev, 3)) {
		aper->ranges[aper->count].base = pci_resource_start(pdev, 3);
		aper->ranges[aper->count].size = pci_resource_len(pdev, 3);
		aper->count++;
	}

	return aper;
}

static int nouveau_remove_conflicting_drivers(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;
	bool primary = false;
	dev_priv->apertures = nouveau_get_apertures(dev);
	if (!dev_priv->apertures)
		return -ENOMEM;

#ifdef CONFIG_X86
	primary = dev->pdev->resource[PCI_ROM_RESOURCE].flags & IORESOURCE_ROM_SHADOW;
#endif
	
	remove_conflicting_framebuffers(dev_priv->apertures, "nouveaufb", primary);
	return 0;
}

extern int pscnv_device_count;
extern struct drm_device **pscnv_drm;

int nouveau_load(struct drm_device *dev, unsigned long flags)
{
	struct drm_nouveau_private *dev_priv;
	uint32_t reg0, strap;
	resource_size_t mmio_start_offs;

	dev_priv = kzalloc(sizeof(*dev_priv), GFP_KERNEL);
	if (!dev_priv)
		return -ENOMEM;
	dev->dev_private = dev_priv;
	dev_priv->dev = dev;

	dev_priv->flags = flags/* & NOUVEAU_FLAGS*/;
	dev_priv->init_state = NOUVEAU_CARD_INIT_DOWN;

	NV_DEBUG(dev, "vendor: 0x%X device: 0x%X class: 0x%X\n",
		 dev->pci_vendor, dev->pci_device, dev->pdev->class);

	dev_priv->wq = create_workqueue("nouveau");
	if (!dev_priv->wq)
		return -EINVAL;

	/* resource 0 is mmio regs */
	/* resource 1 is linear FB */
	/* resource 2 is RAMIN (mmio regs + 0x1000000) */
	/* resource 6 is bios */

	/* map the mmio regs */
	mmio_start_offs = pci_resource_start(dev->pdev, 0);
	dev_priv->mmio = ioremap(mmio_start_offs, 0x00800000);
	if (!dev_priv->mmio) {
		NV_ERROR(dev, "Unable to initialize the mmio mapping. "
			 "Please report your setup to " DRIVER_EMAIL "\n");
		return -EINVAL;
	}
	NV_DEBUG(dev, "regs mapped ok at 0x%llx\n",
					(unsigned long long)mmio_start_offs);

#ifdef __BIG_ENDIAN
	/* Put the card in BE mode if it's not */
	if (nv_rd32(dev, NV03_PMC_BOOT_1))
		nv_wr32(dev, NV03_PMC_BOOT_1, 0x00000001);

	DRM_MEMORYBARRIER();
#endif

	/* Time to determine the card architecture */
	reg0 = nv_rd32(dev, NV03_PMC_BOOT_0);

	/* We're dealing with >=NV10 */
	if ((reg0 & 0x0f000000) > 0) {
		/* Bit 27-20 contain the architecture in hex */
		dev_priv->chipset = (reg0 & 0xff00000) >> 20;
	/* NV04 or NV05 */
	} else if ((reg0 & 0xff00fff0) == 0x20004000) {
		if (reg0 & 0x00f00000)
			dev_priv->chipset = 0x05;
		else
			dev_priv->chipset = 0x04;
	} else {
		dev_priv->chipset = (reg0 & 0xf0000) >> 16;
		if (dev_priv->chipset < 1 || dev_priv->chipset > 3)
			dev_priv->chipset = 0xff;
	}

	switch (dev_priv->chipset & 0xf0) {
	case 0x00:
		if (dev_priv->chipset >= 4)
			dev_priv->card_type = NV_04;
		else
			dev_priv->card_type = dev_priv->chipset;
		break;
	case 0x10:
	case 0x20:
	case 0x30:
		dev_priv->card_type = dev_priv->chipset & 0xf0;
		break;
	case 0x40:
	case 0x60:
		dev_priv->card_type = NV_40;
		break;
	case 0x50:
	case 0x80:
	case 0x90:
	case 0xa0:
		dev_priv->card_type = NV_50;
		break;
	case 0xc0:
		dev_priv->card_type = NV_C0;
		break;
	default:
		NV_INFO(dev, "Unsupported chipset 0x%08x\n", reg0);
		return -EINVAL;
	}

	NV_INFO(dev, "Detected an NV%02x generation card (0x%08x)\n",
		dev_priv->card_type, reg0);

	/* determine frequency of timing crystal */
	strap = nv_rd32(dev, 0x101000);
	if ( dev_priv->chipset < 0x17 ||
	    (dev_priv->chipset >= 0x20 && dev_priv->chipset <= 0x25))
		strap &= 0x00000040;
	else
		strap &= 0x00400040;

	switch (strap) {
	case 0x00000000: dev_priv->crystal = 13500; break;
	case 0x00000040: dev_priv->crystal = 14318; break;
	case 0x00400000: dev_priv->crystal = 27000; break;
	case 0x00400040: dev_priv->crystal = 25000; break;
	}

	NV_DEBUG(dev, "crystal freq: %dKHz\n", dev_priv->crystal);

	dev_priv->fb_size = pci_resource_len(dev->pdev, 1);
	dev_priv->fb_phys = pci_resource_start(dev->pdev, 1);
	dev_priv->mmio_phys = pci_resource_start(dev->pdev, 0);

	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		int ret = nouveau_remove_conflicting_drivers(dev);
		if (ret)
			return ret;
	}

	/* map larger RAMIN aperture on NV40 cards */
	if (dev_priv->card_type >= NV_40) {
		int ramin_bar = 2;
		if (pci_resource_len(dev->pdev, ramin_bar) == 0)
			ramin_bar = 3;

		dev_priv->ramin_size = pci_resource_len(dev->pdev, ramin_bar);
		dev_priv->ramin = ioremap(
				pci_resource_start(dev->pdev, ramin_bar),
				dev_priv->ramin_size);
		if (!dev_priv->ramin) {
			NV_ERROR(dev, "Failed to init RAMIN mapping\n");
			return -ENOMEM;
		}
	}

	nouveau_OF_copy_vbios_to_ramin(dev);

	/* Special flags */
	if (dev->pci_device == 0x01a0)
		dev_priv->flags |= NV_NFORCE;
	else if (dev->pci_device == 0x01f0)
		dev_priv->flags |= NV_NFORCE2;

	/* For kernel modesetting, init card now and bring up fbcon */
	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		int ret = nouveau_card_init(dev);
		if (ret)
			return ret;
	}

	if (dev->primary->index < pscnv_device_count)
		pscnv_drm[dev->primary->index] = dev;

	return 0;
}

static void nouveau_close(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;

	/* In the case of an error dev_priv may not be allocated yet */
	if (dev_priv)
		nouveau_card_takedown(dev);
}

/* KMS: we need mmio at load time, not when the first drm client opens. */
void nouveau_lastclose(struct drm_device *dev)
{
	if (drm_core_check_feature(dev, DRIVER_MODESET))
		return;

	nouveau_close(dev);
}

int nouveau_unload(struct drm_device *dev)
{
	struct drm_nouveau_private *dev_priv = dev->dev_private;

	if (drm_core_check_feature(dev, DRIVER_MODESET)) {
		drm_kms_helper_poll_fini(dev);
		nouveau_fbcon_fini(dev);
		dev_priv->engine.display.destroy(dev);
		nouveau_close(dev);
	}

	iounmap(dev_priv->mmio);
	iounmap(dev_priv->ramin);

	kfree(dev_priv);
	dev->dev_private = NULL;
	return 0;
}

/* Wait until (value(reg) & mask) == val, up until timeout has hit */
bool nouveau_wait_until(struct drm_device *dev, uint64_t timeout,
			uint32_t reg, uint32_t mask, uint32_t val)
{
	uint64_t start = nv04_timer_read(dev);

	do {
		if ((nv_rd32(dev, reg) & mask) == val)
			return true;
	} while (nv04_timer_read(dev) - start < timeout);

	return false;
}

/* Wait until (value(reg) & mask) != val, up until timeout has hit. */
bool nouveau_wait_until_neq(struct drm_device *dev, uint64_t timeout,
			    uint32_t reg, uint32_t mask, uint32_t val)
{
	uint64_t start = nv04_timer_read(dev);

	do {
		if ((nv_rd32(dev, reg) & mask) != val)
			return true;
	} while (nv04_timer_read(dev) - start < timeout);

	return false;
}

bool
nouveau_wait_cb(struct drm_device *dev, uint64_t timeout,
		bool (*cond)(void *), void *data)
{
	uint64_t start = nv04_timer_read(dev);

	do {
		if (cond(data) == true)
			return true;
	} while (nv04_timer_read(dev) - start < timeout);

	return false;
}
