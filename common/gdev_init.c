/*
 * Copyright 2011 Shinpei Kato
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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gdev_init.h"

/* initialize the physical device information. */
int gdev_init_device(struct gdev_device *gdev, int minor, void *priv)
{
	gdev->id = minor;
	gdev->users = 0;
	gdev->mem_size = 0;
	gdev->mem_used = 0;
	gdev->dma_mem_size = 0;
	gdev->dma_mem_used = 0;
	gdev->proc_util = 100; /* 100% */
	gdev->mem_util = 100; /* 100% */
	gdev->swap = NULL;
	gdev->sched_thread = NULL;
	gdev->parent = NULL;
	gdev->priv = priv; /* this must be set before calls to gdev_query(). */
	gdev_list_init(&gdev->vas_list, NULL); /* VAS list. */
	LOCK_INIT(&gdev->vas_lock);
	MUTEX_INIT(&gdev->shmem_mutex);

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

	GDEV_PRINT("Device %d: memory size (device 0x%llx, host 0x%llx)\n",
			   gdev->id, gdev->mem_size, gdev->dma_mem_size);

	return 0;
}

/* finalize the physical device. */
void gdev_exit_device(struct gdev_device *gdev)
{
}

/* initialize the virtual device information. */
int gdev_init_vdevice
(struct gdev_device *gdev, int id, uint32_t proc_util, uint32_t mem_util, 
 struct gdev_device *phys)
{
	gdev->id = id;
	gdev->users = 0;
	gdev->proc_util = proc_util;
	gdev->mem_util = mem_util;
	gdev->swap = NULL;
	gdev->sched_thread = NULL;
	gdev->parent = phys;
	gdev->priv = phys->priv;
	gdev->compute = phys->compute;
	gdev->mem_size = phys->mem_size * mem_util / 100;
	gdev->dma_mem_size = phys->dma_mem_size * mem_util / 100;
	gdev->chipset = phys->chipset;
	gdev_list_init(&gdev->vas_list, NULL); /* VAS list. */
	LOCK_INIT(&gdev->vas_lock);
	MUTEX_INIT(&gdev->shmem_mutex);

	GDEV_PRINT("Virtual device %d: memory size (device 0x%llx, host 0x%llx)\n",
			   gdev->id, gdev->mem_size, gdev->dma_mem_size);

	return 0;
}

/* finalize the virtual device. */
void gdev_exit_vdevice(struct gdev_device *gdev)
{
}
