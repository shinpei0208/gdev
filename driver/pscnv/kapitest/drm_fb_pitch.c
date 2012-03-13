#include "drmP.h"
#include "drm.h"
#include "drm_crtc.h"
#include "drm_crtc_helper.h"
#include "drm_fb_helper.h"

int dummy(struct fb_info *info, struct drm_framebuffer *fb)
{
	drm_fb_helper_fill_fix(info, fb->pitch, fb->depth);
	return 0;
}
