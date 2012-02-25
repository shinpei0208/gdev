#ifndef LIBPSCNV_IB_H
#define LIBPSCNV_IB_H
#include <stdint.h>
#include <sched.h>

struct pscnv_ib_bo {
	int fd;
	int vid;
	uint32_t handle;
	void *map;
	uint32_t size;
	uint64_t vm_base;
};

struct pscnv_ib_chan {
	int fd;
	int vid;
	int cid;
	volatile uint32_t *chmap;

	uint32_t pb_dma;

	struct pscnv_ib_bo *ib;
	uint32_t *ib_map;
	uint32_t ib_order;
	uint32_t ib_mask;
	uint32_t ib_put;
	uint32_t ib_get;

	struct pscnv_ib_bo *pb;
	uint32_t *pb_map;
	uint32_t pb_order;
	uint64_t pb_base;
	uint32_t pb_mask;
	uint32_t pb_size;
	uint32_t pb_pos;
	uint32_t pb_put;
	uint32_t pb_get;

};

int pscnv_ib_chan_new(int fd, int vid, struct pscnv_ib_chan **res, uint32_t pb_dma, uint32_t pb_order, uint32_t ib_order, uint32_t chipset);
int pscnv_ib_bo_alloc(int fd, int vid, uint32_t cookie, uint32_t flags, uint32_t tile_flags, uint64_t size, uint32_t *user, struct pscnv_ib_bo **res);
int pscnv_ib_bo_free(struct pscnv_ib_bo *bo);
int pscnv_ib_push(struct pscnv_ib_chan *ch, uint64_t base, uint32_t len, int flags);
int pscnv_ib_update_get(struct pscnv_ib_chan *ch);

static inline void FIRE_RING(struct pscnv_ib_chan *ch) {
	if (ch->pb_pos != ch->pb_put) {
		if (ch->pb_pos > ch->pb_put) {
			pscnv_ib_push(ch, ch->pb_base + ch->pb_put, ch->pb_pos - ch->pb_put, 0);
		} else {
			pscnv_ib_push(ch, ch->pb_base + ch->pb_put, ch->pb_size - ch->pb_put, 0);
			if (ch->pb_pos)
			       	pscnv_ib_push(ch, ch->pb_base, ch->pb_pos, 0);
		}
		ch->pb_put = ch->pb_pos;
	}
}

static inline void OUT_RING(struct pscnv_ib_chan *ch, uint32_t word) {
	while (((ch->pb_pos + 4) & ch->pb_mask) == ch->pb_get) {
		uint32_t old = ch->pb_get;
		FIRE_RING(ch);
		pscnv_ib_update_get(ch);
		if (old == ch->pb_get)
			sched_yield();
	}
	ch->pb_map[ch->pb_pos/4] = word;
	ch->pb_pos += 4;
	ch->pb_pos &= ch->pb_mask;
}

static inline void BEGIN_RING
(struct pscnv_ib_chan *ch, int subc, int mthd, int len) {
	OUT_RING(ch, mthd | subc << 13 | len << 18);
}

static inline void BEGIN_RING_CONST
(struct pscnv_ib_chan *ch, int subc, int mthd, int len) {
	OUT_RING(ch, mthd | subc << 13 | len << 18 | (0x4 << 28));
}

static inline void BEGIN_RING_NVC0
(struct pscnv_ib_chan *ch, int subc, int mthd, int len) {
	OUT_RING(ch, (0x2 << 28) | (len << 16) | (subc << 13) | (mthd >> 2));
}

static inline void BEGIN_RING_NVC0_CONST
(struct pscnv_ib_chan *ch, int subc, int mthd, int len) {
	OUT_RING(ch, (0x6 << 28) | (len << 16) | (subc << 13) | (mthd >> 2));
}

#endif
