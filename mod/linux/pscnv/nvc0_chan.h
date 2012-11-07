#ifndef __NVC0_CHAN_H__
#define __NVC0_CHAN_H__

#include "drmP.h"
#include "drm.h"
#include "pscnv_chan.h"

#define nvc0_ch(x) container_of(x, struct nvc0_chan_engine, base)

struct nvc0_chan_engine {
	struct pscnv_chan_engine base;
};

#endif /* __NVC0_CHAN_H__ */
