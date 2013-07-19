#ifndef __GDEV_INTERFACE_H__
#define __GDEV_INTERFACE_H__

#include "drmP.h"
#include "drm.h"

#define GDEV_DRV_BO_VRAM 0x1
#define GDEV_DRV_BO_SYSRAM 0x2
#define GDEV_DRV_BO_MAPPABLE 0x4
#define GDEV_DRV_BO_VSPACE 0x8

#define GDEV_DRV_GETPARAM_MP_COUNT 1
#define GDEV_DRV_GETPARAM_FB_SIZE 2
#define GDEV_DRV_GETPARAM_AGP_SIZE 3
#define GDEV_DRV_GETPARAM_CHIPSET_ID 4
#define GDEV_DRV_GETPARAM_BUS_TYPE 5
#define GDEV_DRV_GETPARAM_PCI_VENDOR 6
#define GDEV_DRV_GETPARAM_PCI_DEVICE 7

struct gdev_drv_vspace {
	void *priv;
	void *drm;
};

struct gdev_drv_chan {
	void *priv;
	uint32_t cid;
	volatile uint32_t *regs; /* channel control registers. */
	void *ib_bo; /* driver private object. */
	uint32_t *ib_map;
	uint32_t ib_order;
	uint64_t ib_base;
	uint32_t ib_mask;
	void *pb_bo; /* driver private object. */
	uint32_t *pb_map;
	uint32_t pb_order;
	uint64_t pb_base;
	uint32_t pb_mask;
	uint32_t pb_size;
};

struct gdev_drv_bo {
	void *priv;
	uint64_t addr;
	uint64_t size;
	void *map;
};

int gdev_drv_vspace_alloc(struct drm_device *drm, uint64_t size, struct gdev_drv_vspace *drv_vspace);
int gdev_drv_vspace_free(struct gdev_drv_vspace *drv_vspace);
int gdev_drv_chan_alloc(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_chan *drv_chan);
int gdev_drv_chan_free(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_chan *drv_chan);
int gdev_drv_bo_alloc(struct drm_device *drm, uint64_t size, uint32_t flags, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo);
int gdev_drv_bo_free(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo);
int gdev_drv_bo_bind(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo);
int gdev_drv_bo_unbind(struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo);
int gdev_drv_bo_map(struct drm_device *drm, struct gdev_drv_bo *drv_bo);
int gdev_drv_bo_unmap(struct gdev_drv_bo *drv_bo);
int gdev_drv_read32(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint32_t *p);
int gdev_drv_write32(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint32_t val);
int gdev_drv_read(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t size, void *buf);
int gdev_drv_write(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t size, const void *buf);
int gdev_drv_getdevice(int *count);
int gdev_drv_getdrm(int minor, struct drm_device **pptr);
int gdev_drv_getparam(struct drm_device *drm, uint32_t type, uint64_t *res);
int gdev_drv_getaddr(struct drm_device *drm, struct gdev_drv_vspace *drv_vspace, struct gdev_drv_bo *drv_bo, uint64_t offset, uint64_t *addr);
int gdev_drv_setnotify(void (*func)(int subc, uint32_t data));
int gdev_drv_unsetnotify(void (*func)(int subc, uint32_t data));

#endif
