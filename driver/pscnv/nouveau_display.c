/*
 * Copyright (C) 2008 Maarten Maathuis.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "drmP.h"
#include "drm_crtc_helper.h"
#include "nouveau_drv.h"
#include "nouveau_fb.h"
#include "nouveau_fbcon.h"
#include "pscnv_kapi.h"

static void
nouveau_user_framebuffer_destroy(struct drm_framebuffer *drm_fb)
{
	struct nouveau_framebuffer *fb = nouveau_framebuffer(drm_fb);

	if (fb->nvbo && fb->nvbo->gem)
		drm_gem_object_unreference_unlocked(fb->nvbo->gem);

	drm_framebuffer_cleanup(drm_fb);
	kfree(fb);
}

static int
nouveau_user_framebuffer_create_handle(struct drm_framebuffer *drm_fb,
				       struct drm_file *file_priv,
				       unsigned int *handle)
{
	struct nouveau_framebuffer *fb = nouveau_framebuffer(drm_fb);

	return drm_gem_handle_create(file_priv, fb->nvbo->gem, handle);
}

static const struct drm_framebuffer_funcs nouveau_framebuffer_funcs = {
	.destroy = nouveau_user_framebuffer_destroy,
	.create_handle = nouveau_user_framebuffer_create_handle,
};

#ifdef PSCNV_KAPI_DRM_MODE_FB_CMD2
int
nouveau_framebuffer_init(struct drm_device *dev, struct nouveau_framebuffer *nouveau_fb,
			 struct drm_mode_fb_cmd2 *mode_cmd, struct pscnv_bo *nvbo)
#else
int
nouveau_framebuffer_init(struct drm_device *dev, struct nouveau_framebuffer *nouveau_fb,
			 struct drm_mode_fb_cmd *mode_cmd, struct pscnv_bo *nvbo)
#endif
{
	int ret;

	ret = drm_framebuffer_init(dev, &nouveau_fb->base, &nouveau_framebuffer_funcs);
	if (ret) {
		return ret;
	}

	drm_helper_mode_fill_fb_struct(&nouveau_fb->base, mode_cmd);
	nouveau_fb->nvbo = nvbo;
	return 0;
}

#ifdef PSCNV_KAPI_DRM_MODE_FB_CMD2
static struct drm_framebuffer *
nouveau_user_framebuffer_create(struct drm_device *dev,
				struct drm_file *file_priv,
				struct drm_mode_fb_cmd2 *mode_cmd)
#else
static struct drm_framebuffer *
nouveau_user_framebuffer_create(struct drm_device *dev,
				struct drm_file *file_priv,
				struct drm_mode_fb_cmd *mode_cmd)
#endif
{
	struct nouveau_framebuffer *nouveau_fb;
	struct drm_gem_object *gem;
	int ret;

#ifdef PSCNV_KAPI_DRM_MODE_FB_CMD2
	gem = drm_gem_object_lookup(dev, file_priv, mode_cmd->handles[0]);
#else
	gem = drm_gem_object_lookup(dev, file_priv, mode_cmd->handle);
#endif
	if (!gem)
		return ERR_PTR(-ENOENT);

	nouveau_fb = kzalloc(sizeof(struct nouveau_framebuffer), GFP_KERNEL);
	if (!nouveau_fb)
		return ERR_PTR(-ENOMEM);

	ret = nouveau_framebuffer_init(dev, nouveau_fb, mode_cmd, gem->driver_private);
	if (ret) {
		drm_gem_object_unreference(gem);
		return ERR_PTR(ret);
	}

	return &nouveau_fb->base;
}

const struct drm_mode_config_funcs nouveau_mode_config_funcs = {
	.fb_create = nouveau_user_framebuffer_create,
	.output_poll_changed = nouveau_fbcon_output_poll_changed,
};

