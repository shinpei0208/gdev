#include "drmP.h"
#include "drm_crtc.h"

void dummy(struct drm_connector_funcs *funcs) {
	funcs->detect(0, 0);
}
