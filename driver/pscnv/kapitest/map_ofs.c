#include "drmP.h"

void dummy (struct drm_driver *d) {
	d->get_map_ofs = drm_core_get_map_ofs;
}
