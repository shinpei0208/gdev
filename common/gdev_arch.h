/*
 * Copyright (C) Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
 *
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __GDEV_ARCH_H__
#define __GDEV_ARCH_H__

#include "gdev_conf.h"
#include "gdev_time.h"

/**
 * prototype declarations.
 */
struct gdev_device; 
struct gdev_kernel;

/**
 * Gdev types: they are not exposed to end users.
 */
typedef struct gdev_vas gdev_vas_t;
typedef struct gdev_ctx gdev_ctx_t;
typedef struct gdev_mem gdev_mem_t;

/**
 * architecture-dependent compute functions.
 */
int gdev_compute_setup(struct gdev_device *gdev);
uint32_t gdev_launch(gdev_ctx_t *ctx, struct gdev_kernel *kern);
uint32_t gdev_memcpy(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size);
uint32_t gdev_memcpy_async(gdev_ctx_t *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size);
uint32_t gdev_read32(gdev_mem_t *mem, uint64_t addr);
void gdev_write32(gdev_mem_t *mem, uint64_t addr, uint32_t val);
int gdev_read(gdev_mem_t *mem, void *buf, uint64_t addr, uint32_t size);
int gdev_write(gdev_mem_t *mem, uint64_t addr, const void *buf, uint32_t size);
int gdev_poll(gdev_ctx_t *ctx, uint32_t seq, struct gdev_time *timeout);
int gdev_barrier(struct gdev_ctx *ctx);
int gdev_query(struct gdev_device *gdev, uint32_t type, uint64_t *result);

/**
 * architecture-dependent resource management functions.
 */
struct gdev_device *gdev_dev_open(int minor);
void gdev_dev_close(struct gdev_device *gdev);
gdev_vas_t *gdev_vas_new(struct gdev_device *gdev, uint64_t size, void *handle);
void gdev_vas_free(gdev_vas_t *vas);
gdev_ctx_t *gdev_ctx_new(struct gdev_device *gdev, gdev_vas_t *vas);
void gdev_ctx_free(gdev_ctx_t *ctx);
int gdev_ctx_get_cid(gdev_ctx_t *ctx);
void gdev_block_start(struct gdev_device *gdev);
void gdev_block_end(struct gdev_device *gdev);
void gdev_access_start(struct gdev_device *gdev);
void gdev_access_end(struct gdev_device *gdev);
void gdev_mem_lock(gdev_mem_t *mem);
void gdev_mem_unlock(gdev_mem_t *mem);
void gdev_mem_lock_all(gdev_vas_t *vas);
void gdev_mem_unlock_all(gdev_vas_t *vas);
gdev_mem_t *gdev_mem_alloc(gdev_vas_t *vas, uint64_t size, int type);
gdev_mem_t *gdev_mem_share(gdev_vas_t *vas, uint64_t size);
void gdev_mem_free(gdev_mem_t *mem);
void gdev_mem_gc(gdev_vas_t *vas);
void *gdev_mem_map(gdev_mem_t *mem, uint64_t offset, uint64_t size);
void gdev_mem_unmap(gdev_mem_t *mem);
gdev_mem_t *gdev_mem_lookup_by_addr(gdev_vas_t *vas, uint64_t addr, int type);
gdev_mem_t *gdev_mem_lookup_by_buf(gdev_vas_t *vas, const void *buf, int type);
void *gdev_mem_getbuf(gdev_mem_t *mem);
uint64_t gdev_mem_getaddr(gdev_mem_t *mem);
uint64_t gdev_mem_getsize(gdev_mem_t *mem);
uint64_t gdev_mem_phys_getaddr(gdev_mem_t *mem, uint64_t offset);
int gdev_shm_create(struct gdev_device *gdev, gdev_vas_t *vas, int key, uint64_t size, int flags);
int gdev_shm_destroy_mark(struct gdev_device *gdev, gdev_mem_t *owner);
gdev_mem_t *gdev_shm_attach(gdev_vas_t *vas, gdev_mem_t *mem, uint64_t size);
void gdev_shm_detach(gdev_mem_t *mem);
gdev_mem_t *gdev_shm_lookup(struct gdev_device *gdev, int id);
int gdev_shm_evict_conflict(gdev_ctx_t *ctx, gdev_mem_t *mem);
int gdev_shm_retrieve_swap(gdev_ctx_t *ctx, gdev_mem_t *mem);
int gdev_shm_retrieve_swap_all(gdev_ctx_t *ctx, gdev_vas_t *vas);
int gdev_swap_create(struct gdev_device *gdev, uint32_t size);
void gdev_swap_destroy(struct gdev_device *gdev);

#endif
