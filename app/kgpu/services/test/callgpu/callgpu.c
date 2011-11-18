/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/gfp.h>
#include <linux/kthread.h>
#include <linux/proc_fs.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <asm/page.h>

#include "../../../kgpu/kgpu.h"

/* customized log function */
#define g_log(level, ...) kgpu_do_log(level, "calg2", ##__VA_ARGS__)
#define dbg(...) g_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)

int mycb(struct kgpu_request *req)
{
    g_log(KGPU_LOG_PRINT, "REQ ID: %d, RESP CODE: %d\n",
	   req->id, req->errcode);
    kgpu_vfree(req->in);
    kgpu_free_request(req);
    return 0;
}

static int __init minit(void)
{
    struct kgpu_request* req;
    
    g_log(KGPU_LOG_PRINT, "loaded\n");

    req = kgpu_alloc_request();
    if (!req) {
	g_log(KGPU_LOG_ERROR, "request null\n");
	return 0;
    }
    
    req->in = kgpu_vmalloc(1024*2);
    if (!req->in) {
	g_log(KGPU_LOG_ERROR, "callgpu out of memory\n");
	kgpu_free_request(req);
	return 0;
    }
    req->insize = 1024;
    req->out = (void*)((unsigned long)(req->in)+1024);
    req->outsize = 1024;
    strcpy(req->service_name, "nonexist service");
    req->callback = mycb;

    kgpu_call_async(req);
    
    return 0;
}

static void __exit mexit(void)
{
    g_log(KGPU_LOG_PRINT, "unload\n");
}

module_init(minit);
module_exit(mexit);

MODULE_LICENSE("GPL");
