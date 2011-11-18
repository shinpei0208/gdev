/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */
 
#ifndef __GPUTILS_H__
#define __GPUTILS_H__

#if 0
static void *alloc_dev_mem(unsigned long size) {
    void *h;
    cudaMalloc(&h, size);
    return h;
}

static void free_dev_mem(void *p) {
    cudaFree(p);
}

#define h2dcpy(dst, src, sz) \
    cudaMemcpy((void*)(dst), (void*)(src), (sz), cudaMemcpyHostToDevice)

#define d2hcpy(dst, src, sz) \
    cudaMemcpy((void*)(dst), (void*)(src), (sz), cudaMemcpyDeviceToHost)
#endif

#endif
