/*
 * Copyright (C) 2013 Marcin Kościelnicki <koriakin@0x04.net>
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

#include "nvrm_priv.h"
#include "nvrm_class.h"
#include <stdlib.h>
#include <sys/mman.h>

struct nvrm_vspace *nvrm_vspace_create(struct nvrm_device *dev) {
	struct nvrm_vspace *vas = calloc(sizeof *vas, 1);
	uint64_t limit = 0;
	if (!vas)
		goto out_alloc;
	vas->ctx = dev->ctx;
	vas->dev = dev;
	vas->ovas = nvrm_handle_alloc(vas->ctx);
	vas->odma = nvrm_handle_alloc(vas->ctx);
	if (nvrm_ioctl_create_vspace(vas->dev, vas->dev->odev, vas->ovas, NVRM_CLASS_MEMORY_VM, 0x00010000, &limit, 0))
		goto out_vspace;
	if (nvrm_ioctl_create_dma(vas->ctx, vas->ovas, vas->odma, NVRM_CLASS_DMA_READ, 0x20000000, 0, limit))
		goto out_dma;

	return vas;

out_dma:
	nvrm_ioctl_destroy(vas->ctx, vas->dev->odev, vas->ovas);
out_vspace:
	nvrm_handle_free(vas->ctx, vas->odma);
	nvrm_handle_free(vas->ctx, vas->ovas);
	free(vas);
out_alloc:
	return 0;
}

void nvrm_vspace_destroy(struct nvrm_vspace *vas) {
	nvrm_ioctl_destroy(vas->ctx, vas->ovas, vas->odma);
	nvrm_ioctl_destroy(vas->ctx, vas->dev->odev, vas->ovas);
	nvrm_handle_free(vas->ctx, vas->odma);
	nvrm_handle_free(vas->ctx, vas->ovas);
	free(vas);
}

struct nvrm_bo *nvrm_bo_create(struct nvrm_vspace *vas, uint64_t size, int sysram) {
	struct nvrm_bo *bo = calloc(sizeof *bo, 1);
	if (!bo)
		goto out_alloc;
	bo->ctx = vas->ctx;
	bo->dev = vas->dev;
	bo->vas = vas;
	bo->size = size;
	bo->handle = nvrm_handle_alloc(bo->ctx);
	uint32_t flags1 = sysram ? 0xd001 : 0x1d101;
#if 0
	uint32_t flags2 = sysram ? 0x3a000000 : 0x18000000; /* snooped. */
#else
	uint32_t flags2 = sysram ? 0x5a000000 : 0x18000000;
#endif
	if (nvrm_ioctl_memory(bo->ctx, bo->dev->odev, bo->vas->ovas, bo->handle, flags1, flags2, 0, size))
		goto out_bo;
	if (nvrm_ioctl_vspace_map(bo->ctx, bo->dev->odev, bo->vas->odma, bo->handle, 0, size, &bo->gpu_addr))
		goto out_map;
	return bo;

out_map:
	nvrm_ioctl_destroy(bo->ctx, bo->dev->odev, bo->handle);
out_bo:
	nvrm_handle_free(bo->ctx, bo->handle);
	free(bo);
out_alloc:
	return 0;
}

uint64_t nvrm_bo_gpu_addr(struct nvrm_bo *bo) {
	return bo->gpu_addr;
}

void *nvrm_bo_host_map(struct nvrm_bo *bo) {
	if (bo->mmap)
		return bo->mmap;
	if (nvrm_ioctl_host_map(bo->ctx, bo->dev->osubdev, bo->handle, 0, bo->size, &bo->foffset))
		goto out_host_map;
	void *res = mmap(0, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED, bo->dev->fd, bo->foffset);
	if (res == MAP_FAILED)
		goto out_mmap;
	bo->mmap = res;
	return bo->mmap;

out_mmap:
	nvrm_ioctl_host_unmap(bo->ctx, bo->dev->osubdev, bo->handle, bo->foffset);
out_host_map:
	return 0;
}

void nvrm_bo_host_unmap(struct nvrm_bo *bo) {
	if (bo->mmap) {
		munmap(bo->mmap, bo->size);
		nvrm_ioctl_host_unmap(bo->ctx, bo->dev->osubdev, bo->handle, bo->foffset);
	}
}

void nvrm_bo_destroy(struct nvrm_bo *bo) {
	nvrm_bo_host_unmap(bo);
	nvrm_ioctl_vspace_unmap(bo->ctx, bo->dev->odev, bo->vas->odma, bo->handle, bo->gpu_addr);
	nvrm_ioctl_destroy(bo->ctx, bo->dev->odev, bo->handle);
	nvrm_handle_free(bo->ctx, bo->handle);
	free(bo);
}
