#include "drmP.h"
#include "drm.h"
#include "drm_fb_helper.h"
#include "nouveau_drv.h"

void dummy(struct drm_nouveau_private *dev_priv, struct fb_info *info)
{
	info->apertures = dev_priv->apertures;
}
