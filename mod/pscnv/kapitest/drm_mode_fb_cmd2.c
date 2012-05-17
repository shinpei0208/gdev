#include "drmP.h"
#include "drm.h"
#include "drm_crtc_helper.h"

int dummy(struct drm_framebuffer fb, struct drm_mode_fb_cmd2 mode_cmd)
{
	drm_helper_mode_fill_fb_struct(&fb, &mode_cmd);
	return 0;
}
