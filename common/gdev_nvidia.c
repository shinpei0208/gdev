/*
 * Copyright (C) Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
 *
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gdev_api.h"
#include "gdev_device.h"
#include "gdev_list.h"
#include "gdev_sched.h"
#include "gdev_time.h"

/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_dev_open(int minor)
{
	return gdev_raw_dev_open(minor);
}

/* close the specified Gdev object. */
void gdev_dev_close(struct gdev_device *gdev)
{
	gdev_raw_dev_close(gdev);
}

/* add a new VAS object into the device VAS list. */
static void __gdev_vas_list_add(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;
	
	gdev_lock_save(&gdev->vas_lock, &flags);
	gdev_list_add(&vas->list_entry, &gdev->vas_list);
	gdev_unlock_restore(&gdev->vas_lock, &flags);
}

/* delete the VAS object from the device VAS list. */
static void __gdev_vas_list_del(struct gdev_vas *vas)
{
	struct gdev_device *gdev = vas->gdev;
	unsigned long flags;
	
	gdev_lock_save(&gdev->vas_lock, &flags);
	gdev_list_del(&vas->list_entry);
	gdev_unlock_restore(&gdev->vas_lock, &flags);
}

/* allocate a new virual address space (VAS) object. */
struct gdev_vas *gdev_vas_new(struct gdev_device *gdev, uint64_t size, void *handle)
{
	struct gdev_vas *vas;

	if (!(vas = gdev_raw_vas_new(gdev, size))) {
		return NULL;
	}

	vas->handle = handle;
	vas->gdev = gdev;
	vas->prio = GDEV_PRIO_DEFAULT;
	gdev_list_init(&vas->list_entry, (void *) vas); /* entry to VAS list. */
	gdev_list_init(&vas->mem_list, NULL); /* device memory list. */
	gdev_list_init(&vas->dma_mem_list, NULL); /* host dma memory list. */
	gdev_lock_init(&vas->lock);

	__gdev_vas_list_add(vas);

	return vas;
}

/* free the specified virtual address space object. */
void gdev_vas_free(struct gdev_vas *vas)
{
	__gdev_vas_list_del(vas);
	gdev_raw_vas_free(vas);
}

/* create a new GPU context object. */
struct gdev_ctx *gdev_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas)
{
	struct gdev_ctx *ctx;
	struct gdev_compute *compute = gdev_compute_get(gdev);

	if (!(ctx = gdev_raw_ctx_new(gdev, vas))) {
		return NULL;
	}

	/* save the paraent object. */
	ctx->vas = vas;

	/* initialize the compute-related objects. this must follow ctx_new(). */
	compute->init(ctx);

	return ctx;
}

/* destroy the specified GPU context object. */
void gdev_ctx_free(struct gdev_ctx *ctx)
{
	gdev_raw_ctx_free(ctx);
}

/* get context ID. */
int gdev_ctx_get_cid(struct gdev_ctx *ctx)
{
	return ctx->cid;
}

/* set the flag to block any access to the device. */
void gdev_block_start(struct gdev_device *gdev)
{
	struct gdev_device *phys = gdev_phys_get(gdev);

	/* we have to spin while some context is accessing the GPU. */
retry:
	if (phys) {
		gdev_lock(&phys->global_lock);
		if (phys->accessed || phys->blocked) {
			gdev_unlock(&phys->global_lock);
			SCHED_YIELD();
			goto retry;
		}
		phys->blocked++;
		gdev_unlock(&phys->global_lock);
	}
	else {
		gdev_lock(&gdev->global_lock);
		if (gdev->accessed || gdev->blocked) {
			gdev_unlock(&gdev->global_lock);
			SCHED_YIELD();
			goto retry;
		}
		gdev->blocked++;
		gdev_unlock(&gdev->global_lock);
	}
}

/* clear the flag to unlock any access to the device. */
void gdev_block_end(struct gdev_device *gdev)
{
	struct gdev_device *phys = gdev_phys_get(gdev);

	if (phys) {
		gdev_lock(&phys->global_lock);
		phys->blocked = 0;
		gdev_unlock(&phys->global_lock);
	}
	else {
		gdev_lock(&gdev->global_lock);
		gdev->blocked = 0;
		gdev_unlock(&gdev->global_lock);
	}
}

/* increment the counter for # of contexts accessing the device. */
void gdev_access_start(struct gdev_device *gdev)
{
	struct gdev_device *phys = gdev_phys_get(gdev);
	
retry:
	if (phys) {
		gdev_lock(&phys->global_lock);
		if (phys->blocked) {
			gdev_unlock(&phys->global_lock);
			SCHED_YIELD();
			goto retry;
		}
		phys->accessed++;
		gdev_unlock(&phys->global_lock);
	}
	else {
		gdev_lock(&gdev->global_lock);
		if (gdev->blocked) {
			gdev_unlock(&gdev->global_lock);
			SCHED_YIELD();
			goto retry;
		}
		gdev->accessed++;
		gdev_unlock(&gdev->global_lock);
	}
}

/* decrement the counter for # of contexts accessing the device. */
void gdev_access_end(struct gdev_device *gdev)
{
	struct gdev_device *phys = gdev_phys_get(gdev);

	if (phys) {
		gdev_lock(&phys->global_lock);
		phys->accessed--;
		gdev_unlock(&phys->global_lock);
	}
	else {
		gdev_lock(&gdev->global_lock);
		gdev->accessed--;
		gdev_unlock(&gdev->global_lock);
	}
}
