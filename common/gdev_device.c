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
#include "gdev_sched.h"
#include "gdev_system.h"

int gdev_count = 0; /* # of physical devices. */
int gdev_vcount = 0; /* # of virtual devices. */
struct gdev_device *gdevs = NULL; /* physical devices */
struct gdev_device *gdev_vds = NULL; /* virtual devices */

int VCOUNT_LIST[GDEV_PHYSICAL_DEVICE_MAX_COUNT] = {
	GDEV0_VIRTUAL_DEVICE_COUNT,
	GDEV1_VIRTUAL_DEVICE_COUNT,
	GDEV2_VIRTUAL_DEVICE_COUNT,
	GDEV3_VIRTUAL_DEVICE_COUNT,
	GDEV4_VIRTUAL_DEVICE_COUNT,
	GDEV5_VIRTUAL_DEVICE_COUNT,
	GDEV6_VIRTUAL_DEVICE_COUNT,
	GDEV7_VIRTUAL_DEVICE_COUNT,
};

void __gdev_init_device(struct gdev_device *gdev, int id)
{
	gdev->id = id;
	gdev->users = 0;
	gdev->accessed = 0;
	gdev->blocked = 0;
	gdev->mem_size = 0;
	gdev->mem_used = 0;
	gdev->dma_mem_size = 0;
	gdev->dma_mem_used = 0;
	gdev->chipset = 0;
	gdev->com_bw = 100;
	gdev->mem_bw = 100;
	gdev->mem_sh = 100;
	gdev->com_bw_used = 0;
	gdev->mem_bw_used = 0;
	gdev->period = 0;
	gdev->com_time = 0;
	gdev->mem_time = 0;
	gdev->swap = NULL;
	gdev->sched_com_thread = NULL;
	gdev->sched_mem_thread = NULL;
	gdev->credit_com_thread = NULL;
	gdev->credit_mem_thread = NULL;
	gdev->current_com = NULL;
	gdev->current_mem = NULL;
	gdev->parent = NULL;
	gdev->priv = NULL;
	gdev_time_us(&gdev->credit_com, 0);
	gdev_time_us(&gdev->credit_mem, 0);
	gdev_list_init(&gdev->sched_com_list, NULL);
	gdev_list_init(&gdev->sched_mem_list, NULL);
	gdev_list_init(&gdev->vas_list, NULL);
	gdev_list_init(&gdev->shm_list, NULL);
	gdev_lock_init(&gdev->sched_com_lock);
	gdev_lock_init(&gdev->sched_mem_lock);
	gdev_lock_init(&gdev->vas_lock);
	gdev_lock_init(&gdev->global_lock);
	gdev_mutex_init(&gdev->shm_mutex);
}

/* initialize the physical device information. */
int gdev_init_device(struct gdev_device *gdev, int id, void *priv)
{
	__gdev_init_device(gdev, id);
	gdev->priv = priv; /* this must be set before calls to gdev_query(). */

	/* architecture-dependent chipset. 
	   this call must be prior to the following. */
	gdev_query(gdev, GDEV_QUERY_CHIPSET, (uint64_t*) &gdev->chipset);

	/* device memory size available for users. */
	gdev_query(gdev, GDEV_QUERY_DEVICE_MEM_SIZE, &gdev->mem_size);
	/* FIXME: substract the amount of memory used not for users' data but
	   this shouldn't be hardcoded. */
	gdev->mem_size -= 0xc010000;

	/* host DMA memory size available for users. */
	gdev_query(gdev, GDEV_QUERY_DMA_MEM_SIZE, &gdev->dma_mem_size);

	/* set up the compute engine. */
	gdev_compute_setup(gdev);

	return 0;
}

/* finalize the physical device. */
void gdev_exit_device(struct gdev_device *gdev)
{
}

/* initialize the virtual device information. */
int gdev_init_virtual_device(struct gdev_device *gdev, int id, uint32_t weight, struct gdev_device *phys)
{
	__gdev_init_device(gdev, id);
	gdev->period = GDEV_PERIOD_DEFAULT;
	gdev->parent = phys;
	gdev->priv = gdev_priv_get(gdev_phys_get(gdev));
	gdev->compute = gdev_phys_get(gdev)->compute;
	gdev->mem_size = gdev_phys_get(gdev)->mem_size * weight / 100;
	gdev->dma_mem_size = gdev_phys_get(gdev)->dma_mem_size * weight / 100;
	gdev->com_bw = weight;
	gdev->mem_bw = weight;
	gdev->mem_sh = weight;
	gdev->chipset = gdev_phys_get(gdev)->chipset;

	/* create the swap memory object, if configured, for the virtual device. */
	if (GDEV_SWAP_MEM_SIZE > 0) {
		gdev_swap_create(gdev, GDEV_SWAP_MEM_SIZE);
	}

	return 0;
}

/* finalize the virtual device. */
void gdev_exit_virtual_device(struct gdev_device *gdev)
{
	if (GDEV_SWAP_MEM_SIZE > 0) {
		gdev_swap_destroy(gdev);
	}
}
