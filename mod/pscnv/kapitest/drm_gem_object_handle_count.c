#include "drmP.h"

void dummy(struct drm_gem_object *obj)
{
	atomic_inc(&obj->handle_count);
}
