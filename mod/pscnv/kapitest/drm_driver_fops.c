#include "drmP.h"
#include "drm.h"
#include <linux/module.h>

static struct drm_driver driver = {
	.fops = {
		.owner = THIS_MODULE,
		.open = NULL,
		.release = NULL,
		.unlocked_ioctl = NULL,
		.mmap = NULL,
		.poll = NULL,
		.fasync = NULL,
	},
};

static struct drm_driver* dummy(void)
{
	return &driver;
}
