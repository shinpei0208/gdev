/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Common header for userspace helper, kernel mode KGPU and KGPU clients
 *
 */

#ifndef __KGPU_H__
#define __KGPU_H__

#define TO_UL(v) ((unsigned long)(v))

#define ADDR_WITHIN(pointer, base, size)		\
    (TO_UL(pointer) >= TO_UL(base) &&			\
     (TO_UL(pointer) < TO_UL(base)+TO_UL(size)))

#define ADDR_REBASE(dst_base, src_base, pointer)			\
    (TO_UL(dst_base) + (						\
	TO_UL(pointer)-TO_UL(src_base)))

struct kgpu_gpu_mem_info {
    void *uva;
    unsigned long size;
};

#define KGPU_SERVICE_NAME_SIZE 32

struct kgpu_ku_request {
    int id;
    char service_name[KGPU_SERVICE_NAME_SIZE];
    void *in, *out, *data;
    unsigned long insize, outsize, datasize;
};

/* kgpu's errno */
#define KGPU_OK 0
#define KGPU_NO_RESPONSE 1
#define KGPU_NO_SERVICE 2
#define KGPU_TERMINATED 3

struct kgpu_ku_response {
    int id;
    int errcode;
};

/*
 * Only for kernel code or helper
 */
#if defined __KERNEL__ || defined __KGPU__

/* the NR will not be used */
#define KGPU_BUF_NR 1
#define KGPU_BUF_SIZE (512*1024*1024)

#define KGPU_MMAP_SIZE KGPU_BUF_SIZE

#define KGPU_DEV_NAME "kgpu"

/* ioctl */
#include <linux/ioctl.h>

#define KGPU_IOC_MAGIC 'g'

#define KGPU_IOC_SET_GPU_BUFS \
    _IOW(KGPU_IOC_MAGIC, 1, struct kgpu_gpu_mem_info[KGPU_BUF_NR])
#define KGPU_IOC_GET_GPU_BUFS \
    _IOR(KGPU_IOC_MAGIC, 2, struct kgpu_gpu_mem_info[KGPU_BUF_NR])
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 3)
#define KGPU_IOC_GET_REQS     _IOR(KGPU_IOC_MAGIC, 4, 

#define KGPU_IOC_MAXNR 4

#include "kgpu_log.h"

#endif /* __KERNEL__ || __KGPU__  */

/*
 * For helper and service providers
 */
#ifndef __KERNEL__

struct kgpu_service;

struct kgpu_service_request {
    int id;
    void *hin, *hout, *hdata;
    void *din, *dout, *ddata;
    unsigned long insize, outsize, datasize;
    int errcode;
    struct kgpu_service *s;
    int block_x, block_y;
    int grid_x, grid_y;
    int state;
    int stream_id;
    unsigned long stream;
};

/* service request states: */
#define KGPU_REQ_INIT 1
#define KGPU_REQ_MEM_DONE 2
#define KGPU_REQ_PREPARED 3
#define KGPU_REQ_RUNNING 4
#define KGPU_REQ_POST_EXEC 5
#define KGPU_REQ_DONE 6

#include "service.h"

#endif /* no __KERNEL__ */

/*
 * For kernel code only
 */
#ifdef __KERNEL__

#include <linux/list.h>

struct kgpu_request;

typedef int (*kgpu_callback)(struct kgpu_request *req);

struct kgpu_request {
    int id;
    void *in, *out, *udata, *kdata;
    unsigned long insize, outsize, udatasize, kdatasize;
    char service_name[KGPU_SERVICE_NAME_SIZE];
    kgpu_callback callback;
    int errcode;
};

extern int kgpu_call_sync(struct kgpu_request*);
extern int kgpu_call_async(struct kgpu_request*);

extern int kgpu_next_request_id(void);
extern struct kgpu_request* kgpu_alloc_request(void);
extern void kgpu_free_request(struct kgpu_request*);

extern void *kgpu_vmalloc(unsigned long nbytes);
extern void kgpu_vfree(void* p);

extern void *kgpu_map_pfns(unsigned long *pfns, int n);
extern void *kgpu_map_pages(struct page **pages, int n);
extern void kgpu_unmap_area(unsigned long addr);
extern int kgpu_map_page(struct page*, unsigned long);
extern void kgpu_free_mmap_area(unsigned long);
extern unsigned long kgpu_alloc_mmap_area(unsigned long);

#endif /* __KERNEL__ */

#endif
