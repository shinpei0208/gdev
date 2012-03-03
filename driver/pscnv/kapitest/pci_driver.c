#include "drmP.h"
#include "drm.h"

void dummy(void)
{
	struct drm_driver driver;
	/* driver.pci_driver = 0;*/
	drm_exit(&driver);
}
