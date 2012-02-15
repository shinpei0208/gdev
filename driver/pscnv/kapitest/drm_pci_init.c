#include "drmP.h"
#include "drm.h"

static struct drm_driver driver;
static struct pci_driver nouveau_pci_driver;

static void dummy(void)
{
	drm_pci_init(&driver, &nouveau_pci_driver);
}
