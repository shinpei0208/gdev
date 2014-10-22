/*
 * Copyright (C) Shinpei Kato
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
#include "gdev_io_memcpy.h"
#include "gdev_util.h"
#include "gdev_nvidia.h"
#include "gdev_nvidia_fifo.h"
#include "gdev_nvidia_nve4.h"
#include "xf86drm.h"
#include "xf86drmMode.h"
#include "nouveau_drm.h"
#include "nouveau.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/unistd.h>

#define GDEV_DEVICE_MAX_COUNT 32

struct gdev_device *lgdev; /* local gdev_device structure for user-space scheduling */
struct gdev_nouveau_ctx_objects {
	struct nouveau_object *comp;
	struct nouveau_object *m2mf;
};

void __nouveau_fifo_space(struct gdev_ctx *ctx, uint32_t len)
{
	struct nouveau_pushbuf *push = (struct nouveau_pushbuf *)ctx->pctx;
	nouveau_pushbuf_space(push, len, 0, 0);
}

void __nouveau_fifo_push(struct gdev_ctx *ctx, uint64_t base, uint32_t len, int flags)
{
	struct nouveau_pushbuf *push = (struct nouveau_pushbuf *)ctx->pctx;
	int dwords = len / 4;
	int p = ctx->fifo.pb_put / 4;
	int max = ctx->fifo.pb_size / 4;

	nouveau_pushbuf_space(push, dwords, 0, 0);
	for (;dwords > 0; dwords--) {
		*push->cur++ = ctx->fifo.pb_map[p++];
		if (p >= max) p = 0;
	}
	ctx->fifo.pb_put += len;
	ctx->fifo.pb_put &= ctx->fifo.pb_mask;
	//nouveau_pushbuf_kick(push, push->channel);
}

void __nouveau_fifo_kick(struct gdev_ctx *ctx)
{
	struct nouveau_pushbuf *push = (struct nouveau_pushbuf *)ctx->pctx;
	nouveau_pushbuf_kick(push, push->channel);
}

void __nouveau_fifo_update_get(struct gdev_ctx *ctx)
{
	//printf("FIXME: need to update FIFO GET in a safe manner.");
	//ctx->fifo.pb_get = 0; /* FIXME */
	ctx->fifo.pb_get = ctx->fifo.pb_put;
}

/* query a piece of the device-specific information. */
int gdev_raw_query(struct gdev_device *gdev, uint32_t type, uint64_t *result)
{
	struct nouveau_client *client = (struct nouveau_client *)gdev->priv;
	struct nouveau_device *dev = (struct nouveau_device *)client->device;

	switch (type) {
	case GDEV_NVIDIA_QUERY_MP_COUNT:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_GRAPH_UNITS, result))
			goto fail;
		/**
		 * GRAPH_UNITS query returns the following:
		 * (u32)priv->gpc_nr | (u32)priv->tpc_total << 8 | (u64)priv->rop_nr << 32;
		 */
		*result = (*result &0xffffff00) >> 8;
		/**
		 * In the older nouveau drivers (< 3.10), NOUVEAU_GETPARAM_GRAPH_UNITS meaning is different.
		 * In that case, (*result) becomes 0x0, so fallback to the predefined value 14.
		 */
		if (!*result) {
			if ((gdev->chipset & 0xf0) < 0xe0) {
				*result = 14;
			}
		}
		break;
	case GDEV_QUERY_DEVICE_MEM_SIZE:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_FB_SIZE, result))
			goto fail;
		break;
	case GDEV_QUERY_DMA_MEM_SIZE:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_AGP_SIZE, result))
			goto fail;
		break;
	case GDEV_QUERY_CHIPSET:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_CHIPSET_ID, result))
			goto fail;
		break;
	case GDEV_QUERY_BUS_TYPE:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_BUS_TYPE, result))
			goto fail;
		break;
	case GDEV_QUERY_AGP_SIZE:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_AGP_SIZE, result))
			goto fail;
		break;
	case GDEV_QUERY_PCI_VENDOR:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_PCI_VENDOR, result))
			goto fail;
		break;
	case GDEV_QUERY_PCI_DEVICE:
		if (nouveau_getparam(dev, NOUVEAU_GETPARAM_PCI_DEVICE, result))
			goto fail;
		break;
	default:
		goto fail;
	}

	return 0;

fail:
	GDEV_PRINT("Failed to query %u\n", type);
	return -EINVAL;
}

#define GDEV_DRM_LEGACY_DEV_NAME DRM_DEV_NAME
#define GDEV_DRM_RENDER_DEV_NAME "%s/renderD%d"

static int __gdev_open_by_minor(const char* template, const char* name, int minor)
{
	int fd;
	int matched;
	char buf[64];
	drmVersionPtr version;

	sprintf(buf, template, DRM_DIR_NAME, minor);
	if ((fd = open(buf, O_RDWR, 0)) < 0) {
		return -errno;
	}

	if (!(version = drmGetVersion(fd))) {
		return -ENOSYS;
	}

	matched = strcmp(version->name, name) == 0;
	drmFreeVersion(version);
	if (matched) {
		return fd;
	}

	return -ENODEV;
}

static int __gdev_open_legacy_by_minor(const char* name, int minor)
{
	return __gdev_open_by_minor(GDEV_DRM_LEGACY_DEV_NAME, name, minor);
}

static int __gdev_open_render_by_minor(const char* name, int minor)
{
	return __gdev_open_by_minor(GDEV_DRM_RENDER_DEV_NAME, name, minor);
}

static int __gdev_open_by_ordinal(const char* name, int ordinal)
{
	int count = 0;
	int i;
	/* Opening render nodes renderD%d dev */
	for (i = 64 * 2; i < (64 * 3); ++i) {
		int fd = 0;
		if ((fd = __gdev_open_render_by_minor(name, i)) >= 0) {
			if (count++ == ordinal) {
				return fd;
			}
			close(fd);
		}
	}

	/* Opening legacy card%d dev */
	for (i = 0; i < DRM_MAX_MINOR; ++i) {
		int fd = 0;
		if ((fd = __gdev_open_legacy_by_minor(name, i)) >= 0) {
			if (count++ == ordinal) {
				return fd;
			}
			close(fd);
		}
	}
	return -ENODEV;
}


/* open a new Gdev object associated with the specified device. */
struct gdev_device *gdev_raw_dev_open(int minor)
{
	struct nouveau_device *dev;
	struct nouveau_client *priv;
	struct gdev_device *gdev;
	int major, max;
	int fd = __gdev_open_by_ordinal("nouveau", minor);
	if (fd < 0) {
		goto fail_open_fd;
	}

	if (!gdevs) {
#ifndef GDEV_SCHED_DISABLED
		gdevs = (struct gdev_device *)gdev_attach_shms_dev(GDEV_DEVICE_MAX_COUNT); /* FIXME: constant number   */
		if (!gdevs)
			return NULL;
		minor++;
#else
	    	gdevs = malloc(sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT);
		if (!gdevs)
			return NULL;
		memset(gdevs, 0, sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT);
#endif
	}

	gdev = &gdevs[minor];
	major = 0;
	max = 0;
	while( minor > max + VCOUNT_LIST[major] )
	    max += VCOUNT_LIST[major++];

	if (gdev->users == 0) {
		if (nouveau_device_wrap(fd, 1, &dev))
			goto fail_device;

		if (nouveau_client_new(dev, &priv))
			goto fail_client;
#ifdef GDEV_SCHED_DISABLED
		gdev_init_device(gdev, minor, (void *)priv);
#else
		lgdev = MALLOC(sizeof(*lgdev));
		memset(lgdev, 0, sizeof(*lgdev));
		gdev_init_device(lgdev, major, (void *)priv);
		gdev_init_device(gdevs, major, (void *)priv);
		gdev_init_virtual_device(gdev, minor, 100, (void *)ADDR_SUB(gdev,gdevs));
	}else{
		if (nouveau_device_wrap(fd, 1, &dev))
			goto fail_device;

		if (nouveau_client_new(dev, &priv))
			goto fail_client;
		lgdev = MALLOC(sizeof(*lgdev));
		memset(lgdev, 0, sizeof(*lgdev));
		gdev_init_device(lgdev, major, (void *)priv);
#endif
	}
	gdev->users++;

	return gdev;

fail_client:
	nouveau_device_del(&dev);
fail_device:
	drmClose(fd);
fail_open_fd:
	return NULL;
}

/* close the specified Gdev object. */
void gdev_raw_dev_close(struct gdev_device *gdev)
{
#ifndef GDEV_SCHED_DISABLED
	struct nouveau_client *client = (struct nouveau_client *)lgdev->priv;
#else
	struct nouveau_client *client = (struct nouveau_client *)gdev->priv;
#endif
	struct nouveau_device *dev = (struct nouveau_device *)client->device;
	int i;

	gdev->users--;

	if (gdev->users == 0) {
		gdev_exit_device(gdev);
		nouveau_client_del(&client);
		nouveau_device_del(&dev);

		for (i = 0; i < GDEV_DEVICE_MAX_COUNT; i++) {
			if (gdevs[i].users > 0)
				return;
		}
		FREE(gdevs);
		gdevs = NULL;
	}
}

/* allocate a new virual address space object. 
   pscnv_ib_chan_new() will allocate a channel object, too. */
struct gdev_vas *gdev_raw_vas_new(struct gdev_device *gdev, uint64_t size)
{
	int len;
	void *data;
	struct gdev_vas *vas;
	struct nouveau_object *chan;
#ifndef GDEV_SCHED_DISABLED
	struct nouveau_client *client = (struct nouveau_client *)lgdev->priv;
#else
	struct nouveau_client *client = (struct nouveau_client *)gdev->priv;
#endif
	struct nouveau_device *dev = (struct nouveau_device *)client->device;
	struct nouveau_object *nv = &dev->object;
	/* NvDmaFB = 0xbeef0201, NvDma = 0xbeef0202 */
	struct nv04_fifo nv04_data = { .vram = 0xbeef0201, .gart = 0xbeef0202 };
	struct nvc0_fifo nvc0_data = { };
	uint32_t chipset = gdev->chipset;

#ifndef GDEV_SCHED_DISABLED
	if (!(vas = gdev_attach_shms_vas(0)))
	    	goto fail_vas;
#else
	if (!(vas = malloc(sizeof(*vas))))
	    	goto fail_vas;
	memset(vas, 0, sizeof(*vas));
#endif

	if (chipset < 0xc0) {
		data = &nv04_data;
		len = sizeof(nv04_data);
	} else {
		data = &nvc0_data;
		len = sizeof(nvc0_data);
	}

	/* allocate a new channel.*/
	if (nouveau_object_new(nv, 0, NOUVEAU_FIFO_CHANNEL_CLASS, data, len, &chan))
		goto fail_chan;

	/* private data */
	vas->pvas = (void *)chan;
	/* VAS ID */
	vas->vid = client->id;

	return vas;

fail_chan:
	FREE(vas);
fail_vas:
	return NULL;
}

/* free the specified virtual address space object. */
void gdev_raw_vas_free(struct gdev_vas *vas)
{
	struct nouveau_object *chan = (struct nouveau_object *)vas->pvas;

	nouveau_object_del(&chan);
    	FREE(vas);
}

/* create a new GPU context object. 
   there are not many to do here, as we have already allocated a channel
   object in gdev_vas_new(), i.e., @vas holds it. */
struct gdev_ctx *gdev_raw_ctx_new(struct gdev_device *gdev, struct gdev_vas *vas)
{
	int ret;
	struct gdev_ctx *ctx;
	struct nouveau_bo *push_bo;
	struct nouveau_bo *fence_bo;
	struct nouveau_bo *notify_bo;
	struct nouveau_bo *desc_bo;
	unsigned int push_domain;
	unsigned int fence_domain;
	unsigned int notify_domain;
	unsigned int push_flags;
	unsigned int fence_flags;
	struct nouveau_bufctx *bufctx;
	struct nouveau_pushbuf *push;
	struct nouveau_object *chan = (struct nouveau_object *)vas->pvas;
	struct nouveau_client *client = (struct nouveau_client *)gdev->priv;
	struct nouveau_device *dev = client->device;
	struct gdev_nouveau_ctx_objects *ctx_objects;
	struct nouveau_object *m2mf;
	uint32_t m2mf_class = 0;
#if 1 /* un-necessary */
	struct nouveau_object *comp;
	uint32_t comp_class = 0;
#endif

	if (!(ctx = malloc(sizeof(*ctx))))
		goto fail_ctx;
	memset(ctx, 0, sizeof(*ctx));

	ret = nouveau_pushbuf_new(client, chan, 1, 32 * 1024, true, &push);
	if (ret)
		goto fail_pushbuf;

	ret = nouveau_bufctx_new(client, 1, &bufctx);
	if (ret)
		goto fail_bufctx;

	/* this is redundant against the libdrm_nouveau's private pushbuffers, 
	   but we ensure that we are independent of libdrm_nouveau, which is
	   subject to change in the future. */
	push_domain = NOUVEAU_BO_GART;
	ret = nouveau_bo_new(dev, push_domain | NOUVEAU_BO_MAP, 0, 32 * 1024, NULL, &push_bo);
	if (ret)
		goto fail_push_alloc;

	push_flags = NOUVEAU_BO_RDWR;
	ret = nouveau_bo_map(push_bo, push_flags, client);
	if (ret)
		goto fail_push_map;

	memset(push_bo->map, 0, 32*1024);

	push->user_priv = bufctx;

	/* FIFO push buffer setup. */
	ctx->fifo.pb_order = 15;
	ctx->fifo.pb_map = push_bo->map;
	ctx->fifo.pb_bo = push_bo;
	ctx->fifo.pb_base = push_bo->offset;
	ctx->fifo.pb_mask = (1 << ctx->fifo.pb_order) - 1;
	ctx->fifo.pb_size = (1 << ctx->fifo.pb_order);
	ctx->fifo.pb_pos = ctx->fifo.pb_put = ctx->fifo.pb_get = 0;
	ctx->fifo.space = __nouveau_fifo_space;
	ctx->fifo.push = __nouveau_fifo_push;
	ctx->fifo.kick = __nouveau_fifo_kick;
	ctx->fifo.update_get = __nouveau_fifo_update_get;

	/* FIFO index buffer setup. */
	ctx->fifo.ib_order = 12;
	ctx->fifo.ib_map = NULL;
	ctx->fifo.ib_bo = NULL;
	ctx->fifo.ib_base = 0;
	ctx->fifo.ib_mask = (1 << ctx->fifo.ib_order) - 1;
	ctx->fifo.ib_put = ctx->fifo.ib_get = 0;

	/* FIFO init: already done in gdev_vas_new(). */
	/* FIFO command queue registers: DRM will take care of them */

	/* fence buffer. */
	fence_domain = NOUVEAU_BO_GART;
	ret = nouveau_bo_new(dev, fence_domain | NOUVEAU_BO_MAP, 0, GDEV_FENCE_BUF_SIZE, NULL, &fence_bo);
	if (ret)
		goto fail_fence_alloc;
	fence_flags = NOUVEAU_BO_RDWR;
	ret = nouveau_bo_map(fence_bo, fence_flags, client);
	if (ret)
		goto fail_fence_map;
	memset(fence_bo->map, 0, GDEV_FENCE_BUF_SIZE);
	ctx->fence.bo = fence_bo;
	ctx->fence.map = fence_bo->map;
	ctx->fence.addr = fence_bo->offset;
	ctx->fence.seq = 0;

	/* interrupt buffer. */
	notify_domain = NOUVEAU_BO_VRAM;
	ret = nouveau_bo_new(dev, notify_domain, 0, 8 /* 64 bits */, NULL, &notify_bo);
	if (ret)
		goto fail_notify_alloc;
	ctx->notify.bo = notify_bo;
	ctx->notify.addr = notify_bo->offset;

	/* private data */
	ctx->pctx = (void *)push;
	/* context ID = channel ID. */
	ctx->cid = vas->vid;

	if (!(ctx_objects = malloc(sizeof(*ctx_objects))))
		goto fail_ctx_objects;
	memset(ctx_objects, 0, sizeof(*ctx_objects));

	if ((gdev->chipset & 0xf0) < 0xc0){
		m2mf_class = 0x5039;
		comp_class = 0x50c0;
	}else if ((gdev->chipset & 0xf0) < 0xe0){
		m2mf_class = 0x9039;
		comp_class = 0x90c0;
	}else if ((gdev->chipset & 0xf0) < 0xf0){
	    	m2mf_class = 0xa040; /* NVE0 P2MF  */
		comp_class = 0xa0c0; /* NVE0 COMPUTE */
	}else{
	    	m2mf_class = 0xa140; /* NVF0 P2MF  */
		comp_class = 0xa1c0; /* NVF0 COMPUTE */
	}/* XXX: case of 'NV108'  */
	
	/* allocating PGRAPH context for M2MF */
	if (nouveau_object_new(chan, 0xbeef323f, m2mf_class, NULL, 0, &m2mf))
		goto fail_m2mf;
	ctx_objects->m2mf = m2mf;

	/* allocating PGRAPH context for COMPUTE */
	if (nouveau_object_new(chan, 0xbeef90c0, comp_class, NULL, 0, &comp))
		goto fail_comp;
	ctx_objects->comp = comp;

	ctx->pdata = (void *)ctx_objects;
	
	/* compute desc struct. 
	 * In fact, it must be created for each kernel launch.
	 * need fix.
	 */
	if ((gdev->chipset & 0xf0) >= 0xe0){
	    if( nouveau_bo_new(dev, NOUVEAU_BO_GART | NOUVEAU_BO_MAP, 0, sizeof(struct gdev_nve4_compute_desc), NULL, &desc_bo)){
		goto fail_desc;
	    }
	    ctx->desc.bo = desc_bo;
	    ctx->desc.addr = desc_bo->offset;
	    if(ctx->desc.addr & 0xff || nouveau_bo_map(desc_bo, NOUVEAU_BO_WR, client)){
		goto fail_desc_map;
	    }
	    ctx->desc.map = desc_bo->map;
	    memset(desc_bo->map, 0,sizeof(struct gdev_nve4_compute_desc));
	}else{
	    ctx->desc.bo = NULL;
	}

	nouveau_bufctx_refn(bufctx, 0, push_bo, push_domain | push_flags);
	nouveau_bufctx_refn(bufctx, 0, fence_bo, fence_domain | fence_flags);
	nouveau_bufctx_refn(bufctx, 0, notify_bo, notify_domain | NOUVEAU_BO_RDWR);

	if ((gdev->chipset & 0xf0) >= 0xe0)
	   nouveau_bufctx_refn(bufctx, 0x50/* NVC0_BIND_CP_DESC */, desc_bo, NOUVEAU_BO_GART | NOUVEAU_BO_RD)->priv=NULL;
	
	nouveau_pushbuf_bufctx(push, bufctx);
	nouveau_pushbuf_validate(push);

	return ctx;

fail_desc_map:
	nouveau_bo_ref(NULL, (struct nouveau_bo **)&desc_bo);
fail_desc:
	nouveau_object_del(&comp);
fail_comp:
	nouveau_object_del(&m2mf);
fail_m2mf:
	free(ctx_objects);
fail_ctx_objects:
	nouveau_bo_ref(NULL, &notify_bo);
fail_notify_alloc:
fail_fence_map:
	nouveau_bo_ref(NULL, (struct nouveau_bo **)&fence_bo);
fail_fence_alloc:
	nouveau_bufctx_del(&bufctx);
fail_push_map:
	nouveau_bo_ref(NULL, &push_bo);
fail_bufctx:
fail_push_alloc:
	nouveau_pushbuf_del(&push);
fail_pushbuf:
	free(ctx);
fail_ctx:
	return NULL;
}

/* destroy the specified GPU context object. */
void gdev_raw_ctx_free(struct gdev_ctx *ctx)
{
	struct nouveau_pushbuf *push = (struct nouveau_pushbuf *)ctx->pctx;
	struct nouveau_bufctx *bufctx = (struct nouveau_bufctx *)push->user_priv;
	struct nouveau_bo *push_bo = (struct nouveau_bo *)ctx->fifo.pb_bo;
	struct nouveau_bo *fence_bo = (struct nouveau_bo *)ctx->fence.bo;
	struct nouveau_bo *notify_bo = (struct nouveau_bo *)ctx->notify.bo;
	struct nouveau_bo *desc_bo = (struct nouveau_bo *)ctx->desc.bo;
	struct gdev_nouveau_ctx_objects *ctx_objects = (struct gdev_nouveau_ctx_objects *)ctx->pdata;

	nouveau_bufctx_reset(bufctx, 0);

	if(desc_bo)
	   nouveau_bo_ref(NULL, &desc_bo);
	
	nouveau_bo_ref(NULL, &notify_bo);
	nouveau_bo_ref(NULL, &fence_bo);
	nouveau_bo_ref(NULL, &push_bo);
	nouveau_bufctx_del(&bufctx);
	nouveau_pushbuf_del(&push);
	/*
	 * Since it calls FIFO's context_detach, it kills the channel in this
	 * routine. And later the channel's destructor creates a fence to
	 * synchronize its execution. Then a page fault occurs.
	 * In the meantime, we comment out this compute engine's destruction.
	 * Maybe we should change comp's parent from FIFO to something.
	 * We need to investigate it later.
	 */
	/* nouveau_object_del(&ctx_objects->comp); */
	nouveau_object_del(&ctx_objects->m2mf);
	free(ctx_objects);
	free(ctx);
}

/* allocate a new memory object. */
static struct gdev_mem *__gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t size, uint32_t flags)
{
	struct gdev_mem *mem;
	struct gdev_device *gdev = vas->gdev;
	struct nouveau_client *client = (struct nouveau_client *)gdev->priv;
	struct nouveau_device *dev = client->device;
	struct nouveau_bo *bo;
	
#ifndef GDEV_SCHED_DISABLED
	if (!(mem = (struct gdev_mem *)gdev_attach_shms_mem(0)))
		goto fail_mem;
#else
	if (!(mem = (struct gdev_mem *) malloc(sizeof(*mem))))
		goto fail_mem;
	memset(mem, 0, sizeof(*mem));
#endif
	if (nouveau_bo_new(dev, flags, 0, size, NULL, &bo))
		goto fail_bo;

	/* address, size, and map. */
	mem->addr = bo->offset;
	mem->size = bo->size;
	if (flags & NOUVEAU_BO_MAP) {
		if (nouveau_bo_map(bo, NOUVEAU_BO_RDWR, client))
			goto fail_map;
		mem->map = bo->map;
	}
	else
		mem->map = NULL;

	/* private data. */
	mem->bo = (void *)bo;

	return mem;

fail_map:
	nouveau_bo_ref(NULL, &bo);
fail_bo:
	GDEV_PRINT("Failed to allocate NVI buffer object.\n");
	free(mem);
fail_mem:
	return NULL;
}

/* allocate a new device memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc(struct gdev_vas *vas, uint64_t size)
{
	uint32_t flags = NOUVEAU_BO_VRAM;

	if (size <= GDEV_MEM_MAPPABLE_LIMIT)
		flags |= NOUVEAU_BO_MAP;

	return __gdev_raw_mem_alloc(vas, size, flags);
}

/* allocate a new host DMA memory object. size may be aligned. */
struct gdev_mem *gdev_raw_mem_alloc_dma(struct gdev_vas *vas, uint64_t size)
{
	uint32_t flags = NOUVEAU_BO_GART | NOUVEAU_BO_MAP;
	return __gdev_raw_mem_alloc(vas, size, flags);
}

/* free the specified memory object. */
void gdev_raw_mem_free(struct gdev_mem *mem)
{
	struct nouveau_bo *bo = (struct nouveau_bo *)mem->bo;

	nouveau_bo_ref(NULL, (struct nouveau_bo **)&bo);
#ifdef GDEV_SCHED_DISABLED
	free(mem);
#endif
}

/* allocate a reserved swap memory object. size may be aligned. */
struct gdev_mem *gdev_raw_swap_alloc(struct gdev_device *gdev, uint64_t size)
{
	GDEV_PRINT("Swap memory not implemented\n");
	/* To be implemented. */
	return NULL;
}

/* free the specified swap memory object. */
void gdev_raw_swap_free(struct gdev_mem *mem)
{
	GDEV_PRINT("Swap memory not implemented\n");
	/* To be implemented. */
}

/* create a new memory object sharing memory space with @mem. */
struct gdev_mem *gdev_raw_mem_share(struct gdev_vas *vas, struct gdev_mem *mem)
{
	GDEV_PRINT("Shared memory not implemented\n");
	/* To be implemented. */
	return NULL;
}

/* destroy the memory object by just unsharing memory space. */
void gdev_raw_mem_unshare(struct gdev_mem *mem)
{
	GDEV_PRINT("Shared memory not implemented\n");
	/* To be implemented. */
}

uint32_t gdev_raw_read32(struct gdev_mem *mem, uint64_t addr)
{
	struct nouveau_bo *bo = mem->bo;
	uint64_t offset = addr - bo->offset;
	uint32_t val;

	if (bo->map) {
		val = *(uint32_t*)(bo->map + offset);
	}
	else {
		/* libnvi doesn't support direct device read. */
		val = 0;
	}

	return val;
}

void gdev_raw_write32(struct gdev_mem *mem, uint64_t addr, uint32_t val)
{
	struct nouveau_bo *bo = mem->bo;
	uint64_t offset = addr - bo->offset;

	if (bo->map) {
		*(uint32_t*)(bo->map + offset) = val;
	}
	else {
		/* libnvi doesn't support direct device read. */
	}
}

int gdev_raw_read(struct gdev_mem *mem, void *buf, uint64_t addr, uint32_t size)
{
	struct nouveau_bo *bo = mem->bo;
	uint64_t offset = addr - bo->offset;

	if (bo->map) {
		gdev_io_memcpy(buf, bo->map + offset, size);
		return 0;
	}
	else {
		/* nouveau doesn't support direct device read. */
		return -EINVAL;
	}
}

int gdev_raw_write(struct gdev_mem *mem, uint64_t addr, const void *buf, uint32_t size)
{
	struct nouveau_bo *bo = mem->bo;
	uint64_t offset = addr - bo->offset;

	if (bo->map) {
		gdev_io_memcpy(bo->map + offset, buf, size);
		return 0;
	}
	else {
		/* nouveau doesn't support direct device read. */
		return -EINVAL;
	}
}

/* map device memory to host DMA memory. */
void *gdev_raw_mem_map(struct gdev_mem *mem)
{
	struct nouveau_bo *bo = mem->bo;
	/* with nouveau we suppose all memory objects to be mapped on the host. */
	return bo->map;
}

/* unmap device memory from host DMA memory. */
void gdev_raw_mem_unmap(struct gdev_mem *mem, void *map)
{
	/* with libnvi, nothing to do really. */
}

/* get physical bus address. */
uint64_t gdev_raw_mem_phys_getaddr(struct gdev_mem *mem, uint64_t offset)
{
	struct nouveau_bo *bo = mem->bo;
	return bo->handle; /* FIXME */
}
