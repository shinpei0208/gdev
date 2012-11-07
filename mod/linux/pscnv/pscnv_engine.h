#ifndef __PSCNV_ENGINE_H__
#define __PSCNV_ENGINE_H__

struct pscnv_chan;
struct pscnv_vspace;

struct pscnv_engine {
	struct drm_device *dev;
	uint32_t *oclasses;
	void (*takedown) (struct pscnv_engine *eng);
	int (*tlb_flush) (struct pscnv_engine *eng, struct pscnv_vspace *vs);
	int (*chan_alloc) (struct pscnv_engine *eng, struct pscnv_chan *ch);
	void (*chan_free) (struct pscnv_engine *eng, struct pscnv_chan *ch);
	int (*chan_obj_new) (struct pscnv_engine *eng, struct pscnv_chan *ch, uint32_t handle, uint32_t oclass, uint32_t flags);
	void (*chan_kill) (struct pscnv_engine *eng, struct pscnv_chan *ch);
};

int nv50_graph_init(struct drm_device *dev);
int nvc0_graph_init(struct drm_device *dev);
int nv84_crypt_init(struct drm_device *dev);
int nv98_crypt_init(struct drm_device *dev);
int nvc0_copy_init(struct drm_device *dev, int engine);

#define PSCNV_ENGINE_GRAPH	1
#define PSCNV_ENGINE_COPY	3
#define PSCNV_ENGINE_COPY0	3 /* PSCNV_ENGINE_COPY + 0 */
#define PSCNV_ENGINE_COPY1	4 /* PSCNV_ENGINE_COPY + 1 */
#define PSCNV_ENGINE_CRYPT	5

#define PSCNV_ENGINES_NUM	16


#endif
