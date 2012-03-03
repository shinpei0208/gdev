#ifndef __NV50_VM_H__
#define __NV50_VM_H__

#include "drmP.h"
#include "drm.h"
#include "pscnv_vm.h"

#define NV50_VM_SIZE		0x10000000000ULL
#define NV50_VM_PDE_COUNT	0x800
#define NV50_VM_SPTE_COUNT	0x20000
#define NV50_VM_LPTE_COUNT	0x2000

#define nv50_vm(x) container_of(x, struct nv50_vm_engine, base)
#define nv50_vs(x) ((struct nv50_vspace *)(x)->engdata)

struct nv50_vm_engine {
	struct pscnv_vm_engine base;
	struct pscnv_vspace *barvm;
	struct pscnv_chan *barch;
};

struct nv50_vspace {
	struct list_head chan_list;
	int engref[PSCNV_ENGINES_NUM];
	struct pscnv_bo *pt[NV50_VM_PDE_COUNT];
};

int nv50_vm_flush (struct drm_device *dev, int unit);
void nv50_vm_trap(struct drm_device *dev);

#endif /* __NV50_VM_H__ */
