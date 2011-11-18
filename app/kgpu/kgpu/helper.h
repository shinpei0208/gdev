/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */
 
#ifndef __HELPER_H__
#define __HELPER_H__

#include "kgpu.h"

#define kh_log(level, ...) kgpu_do_log(level, "helper", ##__VA_ARGS__)
#define dbg(...) kh_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)

extern struct kgpu_gpu_mem_info hostbuf;
extern struct kgpu_gpu_mem_info hostvma;
extern struct kgpu_gpu_mem_info devbuf;
extern struct kgpu_gpu_mem_info devbuf4vma;

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#define round_down(x, y) ((x) & ~__round_mask(x, y))

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

#ifdef __cplusplus
extern "C" {
#endif

    void gpu_init(void);
    void gpu_finit(void);

    void *gpu_alloc_pinned_mem(unsigned long size);
    void gpu_free_pinned_mem(void *p);

    void gpu_pin_mem(void *p, size_t sz);
    void gpu_unpin_mem(void *p);

    int gpu_alloc_device_mem(struct kgpu_service_request *sreq);
    void gpu_free_device_mem(struct kgpu_service_request *sreq);

    int gpu_execution_finished(struct kgpu_service_request *sreq);
    int gpu_post_finished(struct kgpu_service_request *sreq);

#ifdef __cplusplus
}
#endif
   
#endif
