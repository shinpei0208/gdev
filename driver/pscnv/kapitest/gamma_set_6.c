#include "drm.h"
#include "drmP.h"
#include "drm_crtc.h"

void dummy(struct drm_crtc_funcs *funcs) {
	funcs->gamma_set(0, 0, 0, 0, 0, 0);
}
