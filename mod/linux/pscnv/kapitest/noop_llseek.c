#include "drmP.h"
#include "drm.h"

static const struct file_operations nouveau_driver_fops = {
	.llseek = noop_llseek,
};

static struct file_operations* dummy(void)
{
	return &nouveau_driver_fops;
}
