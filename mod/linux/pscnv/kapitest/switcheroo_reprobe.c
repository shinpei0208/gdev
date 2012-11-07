#include "drmP.h"
#include "drm.h"
#include <linux/vga_switcheroo.h>

void dummy(struct drm_device *dev)
{
	vga_switcheroo_register_client(dev->pdev, 0, 0);
}
