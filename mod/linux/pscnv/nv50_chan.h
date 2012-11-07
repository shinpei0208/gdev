#ifndef __NV50_CHAN_H__
#define __NV50_CHAN_H__

#include "drmP.h"
#include "drm.h"
#include "pscnv_chan.h"

#define NV50_CHAN_PD	0x1400
#define NV84_CHAN_PD	0x0200

#define nv50_ch(x) container_of(x, struct nv50_chan_engine, base)

struct nv50_chan_engine {
	struct pscnv_chan_engine base;
};

extern int nv50_chan_iobj_new(struct pscnv_chan *, uint32_t size);
extern int nv50_chan_dmaobj_new(struct pscnv_chan *, uint32_t type, uint64_t start, uint64_t size);

#endif /* __NV50_CHAN_H__ */
