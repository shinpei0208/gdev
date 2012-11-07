#include "drmP.h"
#include "drm.h"

static struct drm_driver driver;

static int dummy(struct pci_dev *pdev, const struct pci_device_id *ent)
{
	return drm_get_dev(pdev, ent, &driver);
}
