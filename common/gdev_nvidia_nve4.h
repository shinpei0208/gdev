#ifndef __GDEV_NVIDIA_NVE4_H__
#define __GDEV_NVIDIA_NVE4_H__

#include "gdev_device.h"
#include "gdev_conf.h"

struct gdev_nve4_compute_desc{
	uint32_t unk0[8];
	uint32_t entry;
	uint32_t unk9[3];
	uint32_t griddim_x :31;
	uint32_t unk12 :1;
	uint16_t griddim_y;
	uint16_t griddim_z;
	uint32_t unk14[3];
	uint16_t shared_size; /* must be aligned to 0x100  */
	uint16_t unk15;
	uint16_t unk16;
	uint16_t blockdim_x;
	uint16_t blockdim_y;
	uint16_t blockdim_z;
	uint32_t cb_mask      : 8;
	uint32_t unk20_8      : 21;
	uint32_t cache_split  : 2;
	uint32_t unk20_31     : 1;
	uint32_t unk21[8];
	struct {
	    uint32_t address_l;
	    uint32_t address_h : 8;
	    uint32_t reserved  : 7;
	    uint32_t size      : 17;
	} cb[8];
	uint32_t local_size_p : 20;
	uint32_t unk45_20     : 7;
	uint32_t bar_alloc    : 5;
	uint32_t local_size_n : 20;
	uint32_t unk46_20     : 4;
	uint32_t gpr_alloc    : 8;
	uint32_t cstack_size  : 20;
	uint32_t unk47_20     : 12;
	uint32_t unk48[16];
};

#define GDEV_SUBCH_NV_P2MF GDEV_SUBCH_NV_M2MF

struct gdev_nve4_compute_desc *gdev_nve4_compute_desc_set(struct gdev_ctx *ctx, struct gdev_nve4_compute_desc *desc, struct gdev_kernel *k);
#endif
