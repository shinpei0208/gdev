#ifndef PSCNV_MM_H
#define PSCNV_MM_H

#include "drmP.h"
#include "drm.h"
#include "pscnv_tree.h"

PSCNV_RB_HEAD(pscnv_mm_head, pscnv_mm_node);

struct pscnv_mm {
	struct drm_device *dev;
	struct pscnv_mm_head head;
	uint32_t spsize;
	uint32_t lpsize;
	uint32_t tssize;
};

struct pscnv_mm_node {
	PSCNV_RB_ENTRY(pscnv_mm_node) entry;
	struct pscnv_mm *mm;
	uint64_t maxgap[4];
	uint64_t gap[4];
	int sentinel;
	enum {
		PSCNV_MM_TYPE_USED0,
		PSCNV_MM_TYPE_USED1,
		PSCNV_MM_TYPE_FREE,
	} type;
	uint64_t start;
	uint64_t size;
	struct pscnv_mm_node *next;
	struct pscnv_mm_node *prev;
	void *tag;
	void *tag2;
};

#define PSCNV_MM_T1		1
#define PSCNV_MM_LP		2
#define PSCNV_MM_FRAGOK		4
#define PSCNV_MM_FROMBACK	8

int pscnv_mm_init(struct drm_device *dev, uint64_t start, uint64_t end, uint32_t spsize, uint32_t lpsize, uint32_t tssize, struct pscnv_mm **res);
int pscnv_mm_alloc(struct pscnv_mm *mm, uint64_t size, uint32_t flags, uint64_t start, uint64_t end, struct pscnv_mm_node **res);
void pscnv_mm_free(struct pscnv_mm_node *node);
void pscnv_mm_takedown(struct pscnv_mm *mm, void (*free_callback)(struct pscnv_mm_node *));
struct pscnv_mm_node *pscnv_mm_find_node(struct pscnv_mm *mm, uint64_t addr);

#endif
