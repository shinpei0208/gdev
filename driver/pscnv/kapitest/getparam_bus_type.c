#include "drmP.h"
#include "drm.h"

void dummy(struct drm_device *dev)
{
	drm_device_is_agp(dev);
	drm_device_is_pcie(dev);
}
