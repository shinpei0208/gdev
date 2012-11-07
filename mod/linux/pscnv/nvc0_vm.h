#ifndef __NVC0_VM_H__
#define __NVC0_VM_H__

#include "drmP.h"
#include "drm.h"
#include "pscnv_engine.h"

#define NVC0_VM_SIZE		0x10000000000ULL

#define NVC0_SPAGE_SHIFT        12
#define NVC0_LPAGE_SHIFT        17
#define NVC0_SPAGE_MASK         0x00fff
#define NVC0_LPAGE_MASK         0x1ffff

#define NVC0_VM_PDE_COUNT       0x2000
#define NVC0_VM_BLOCK_SIZE      0x8000000
#define NVC0_VM_BLOCK_MASK      0x7ffffff
#define NVC0_VM_SPTE_COUNT      (NVC0_VM_BLOCK_SIZE >> NVC0_SPAGE_SHIFT)
#define NVC0_VM_LPTE_COUNT      (NVC0_VM_BLOCK_SIZE >> NVC0_LPAGE_SHIFT)

#define NVC0_PDE(a)             ((a) / NVC0_VM_BLOCK_SIZE)
#define NVC0_SPTE(a)            (((a) & NVC0_VM_BLOCK_MASK) >> NVC0_SPAGE_SHIFT)
#define NVC0_LPTE(a)            (((a) & NVC0_VM_BLOCK_MASK) >> NVC0_LPAGE_SHIFT)

#define NVC0_PDE_HT_SIZE 32
#define NVC0_PDE_HASH(n) (n % NVC0_PDE_HT_SIZE)

#define nvc0_vm(x) container_of(x, struct nvc0_vm_engine, base)
#define nvc0_vs(x) ((struct nvc0_vspace *)(x)->engdata)

struct nvc0_pgt {
	struct list_head head;
	unsigned int pde;
	unsigned int limit; /* virtual range = NVC0_VM_BLOCK_SIZE >> limit */
	struct pscnv_bo *bo[2]; /* 128 KiB and 4 KiB page tables */
};

struct nvc0_vm_engine {
	struct pscnv_vm_engine base;
	struct pscnv_vspace *bar1vm;
	struct pscnv_chan *bar1ch;
	struct pscnv_vspace *bar3vm;
	struct pscnv_chan *bar3ch;
};

struct nvc0_vspace {
	struct pscnv_bo *pd;
	struct list_head ptht[NVC0_PDE_HT_SIZE];
	struct pscnv_mm_node *obj19848;
	struct pscnv_mm_node *obj08004;
	struct pscnv_mm_node *obj0800c;
	struct pscnv_bo *mmio_bo;
	struct pscnv_mm_node *mmio_vm;
	uint32_t mmio_count;
};

#endif /* __NVC0_VM_H__ */
