#ifndef NVRM_H
#define NVRM_H

#include <stdint.h>

struct nvrm_context;
struct nvrm_device;
struct nvrm_vspace;
struct nvrm_bo;
struct nvrm_channel;
struct nvrm_eng;

struct nvrm_context *nvrm_open();
void nvrm_close(struct nvrm_context *ctx);

int nvrm_num_devices(struct nvrm_context *ctx);
struct nvrm_device *nvrm_device_open(struct nvrm_context *ctx, int idx);
void nvrm_device_close(struct nvrm_device *dev);
int nvrm_device_get_chipset(struct nvrm_device *dev, uint32_t *major, uint32_t *minor, uint32_t *stepping);

struct nvrm_vspace *nvrm_vspace_create(struct nvrm_device *dev);
void nvrm_vspace_destroy(struct nvrm_vspace *vas);

struct nvrm_bo *nvrm_bo_create(struct nvrm_vspace *vas, uint64_t size, int sysram);
void nvrm_bo_destroy(struct nvrm_bo *bo);
void *nvrm_bo_host_map(struct nvrm_bo *bo);
uint64_t nvrm_bo_gpu_addr(struct nvrm_bo *bo);
void nvrm_bo_host_unmap(struct nvrm_bo *bo);

struct nvrm_channel *nvrm_channel_create_ib(struct nvrm_vspace *vas, uint32_t cls, struct nvrm_bo *ib);
void nvrm_channel_destroy(struct nvrm_channel *chan);
void *nvrm_channel_host_map_regs(struct nvrm_channel *chan);
void *nvrm_channel_host_map_errnot(struct nvrm_channel *chan);

struct nvrm_eng *nvrm_eng_create(struct nvrm_channel *chan, uint32_t eid, uint32_t cls);

#endif
