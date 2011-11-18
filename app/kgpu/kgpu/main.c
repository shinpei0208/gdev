/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU k-u communication module.
 *
 * Weibin Sun
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/gfp.h>
#include <linux/kthread.h>
#include <linux/proc_fs.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/bitmap.h>
#include <linux/device.h>
#include <asm/atomic.h>
#include "kkgpu.h"

struct _kgpu_mempool {
    unsigned long uva;
    unsigned long kva;
    struct page **pages;
    u32 npages;
    u32 nunits;
    unsigned long *bitmap;
    u32           *alloc_sz;
};

struct _kgpu_vma {
    struct vm_area_struct *vma;
    unsigned long start;
    unsigned long end;
    u32 npages;
    u32 *alloc_sz;
    unsigned long *bitmap;
};

struct _kgpu_dev {
    struct cdev cdev;
    struct class *cls;
    dev_t devno;

    int rid_sequence;
    spinlock_t ridlock;

    struct list_head reqs;
    spinlock_t reqlock;
    wait_queue_head_t reqq;

    struct list_head rtdreqs;
    spinlock_t rtdreqlock;

    struct _kgpu_mempool gmpool;
    spinlock_t gmpool_lock;

    struct _kgpu_vma vm;
    spinlock_t vm_lock;
    int state;
};

struct _kgpu_request_item {
    struct list_head list;
    struct kgpu_request *r;
};

struct _kgpu_sync_call_data {
	wait_queue_head_t queue;
	void* oldkdata;
	kgpu_callback oldcallback;
	int done;
};

static atomic_t kgpudev_av = ATOMIC_INIT(1);
static struct _kgpu_dev kgpudev;

static struct kmem_cache *kgpu_request_cache;
static struct kmem_cache *kgpu_request_item_cache;
static struct kmem_cache *kgpu_sync_call_data_cache;

/*
 * Async GPU call.
 */
int kgpu_call_async(struct kgpu_request *req)
{
    struct _kgpu_request_item *item;
    
    if (unlikely(kgpudev.state == KGPU_TERMINATED)) {
	kgpu_log(KGPU_LOG_ALERT,
		 "kgpu is terminated, no request accepted any more\n");
	return KGPU_TERMINATED;
    }
    
    item = kmem_cache_alloc(kgpu_request_item_cache, GFP_KERNEL);

    if (!item) {
	kgpu_log(KGPU_LOG_ERROR, "out of memory for kgpu request\n");
	return -ENOMEM;
    }
    item->r = req;
    
    spin_lock(&(kgpudev.reqlock));

    INIT_LIST_HEAD(&item->list);
    list_add_tail(&item->list, &(kgpudev.reqs));
    
    wake_up_interruptible(&(kgpudev.reqq));
    
    spin_unlock(&(kgpudev.reqlock));

    return 0;
}
EXPORT_SYMBOL_GPL(kgpu_call_async);

/*
 * Callback for sync GPU call.
 */
static int sync_callback(struct kgpu_request *req)
{
    struct _kgpu_sync_call_data *data = (struct _kgpu_sync_call_data*)
	req->kdata;
    
    data->done = 1;
    
    wake_up_interruptible(&data->queue);
    
    return 0;
}

/*
 * Sync GPU call
 */
int kgpu_call_sync(struct kgpu_request *req)
{
    struct _kgpu_sync_call_data *data;
    struct _kgpu_request_item *item;

    if (unlikely(kgpudev.state == KGPU_TERMINATED)) {
	kgpu_log(KGPU_LOG_ALERT,
		 "kgpu is terminated, no request accepted any more\n");
	return KGPU_TERMINATED;
    }
    
    data = kmem_cache_alloc(kgpu_sync_call_data_cache, GFP_KERNEL);
    if (!data) {
	kgpu_log(KGPU_LOG_ERROR, "kgpu_call_sync alloc mem failed\n");
	return -ENOMEM;
    }
    item = kmem_cache_alloc(kgpu_request_item_cache, GFP_KERNEL);
    if (!item) {
	kgpu_log(KGPU_LOG_ERROR, "out of memory for kgpu request\n");
	return -ENOMEM;
    }
    item->r = req;
    
    data->oldkdata = req->kdata;
    data->oldcallback = req->callback;
    data->done = 0;
    init_waitqueue_head(&data->queue);
    
    req->kdata = data;
    req->callback = sync_callback;
    
    spin_lock(&(kgpudev.reqlock));

    INIT_LIST_HEAD(&item->list);
    list_add_tail(&item->list, &(kgpudev.reqs));

    wake_up_interruptible(&(kgpudev.reqq));
    
    spin_unlock(&(kgpudev.reqlock));

    wait_event_interruptible(data->queue, (data->done==1));
        
    req->kdata = data->oldkdata;
    req->callback = data->oldcallback;
    kmem_cache_free(kgpu_sync_call_data_cache, data);
    
    return 0;
}
EXPORT_SYMBOL_GPL(kgpu_call_sync);


int kgpu_next_request_id(void)
{
    int rt = -1;
    
    spin_lock(&(kgpudev.ridlock));
    
    kgpudev.rid_sequence++;
    if (kgpudev.rid_sequence < 0)
	kgpudev.rid_sequence = 0;
    rt = kgpudev.rid_sequence;
    
    spin_unlock(&(kgpudev.ridlock));

    return rt;
}
EXPORT_SYMBOL_GPL(kgpu_next_request_id);

static void kgpu_request_item_constructor(void *data)
{
    struct _kgpu_request_item *item =
	(struct _kgpu_request_item*)data;

    if (item) {
	memset(item, 0, sizeof(struct _kgpu_request_item));
	INIT_LIST_HEAD(&item->list);
	item->r = NULL;
    }
}

static void kgpu_sync_call_data_constructor(void *data)
{
    struct _kgpu_sync_call_data *d =
	(struct _kgpu_sync_call_data*)data;
    if (d) {
	memset(d, 0, sizeof(struct _kgpu_sync_call_data));
    }
}

static void kgpu_request_constructor(void* data)
{
    struct kgpu_request *req = (struct kgpu_request*)data;
    if (req) {
	memset(req, 0, sizeof(struct kgpu_request));
	req->id = kgpu_next_request_id();
	req->service_name[0] = 0;
    }
}

struct kgpu_request* kgpu_alloc_request(void)
{
    struct kgpu_request *req =
	kmem_cache_alloc(kgpu_request_cache, GFP_KERNEL);
    return req;
}
EXPORT_SYMBOL_GPL(kgpu_alloc_request);


void kgpu_free_request(struct kgpu_request* req)
{
    kmem_cache_free(kgpu_request_cache, req);
}
EXPORT_SYMBOL_GPL(kgpu_free_request);


void* kgpu_vmalloc(unsigned long nbytes)
{
    unsigned int req_nunits = DIV_ROUND_UP(nbytes, KGPU_BUF_UNIT_SIZE);
    void *p = NULL;
    unsigned long idx;

    spin_lock(&kgpudev.gmpool_lock);
    idx = bitmap_find_next_zero_area(
	kgpudev.gmpool.bitmap, kgpudev.gmpool.nunits, 0, req_nunits, 0);
    if (idx < kgpudev.gmpool.nunits) {
	bitmap_set(kgpudev.gmpool.bitmap, idx, req_nunits);
	p = (void*)((unsigned long)(kgpudev.gmpool.kva)
		    + idx*KGPU_BUF_UNIT_SIZE);
	kgpudev.gmpool.alloc_sz[idx] = req_nunits;
    } else {
	kgpu_log(KGPU_LOG_ERROR, "out of GPU memory for malloc %lu\n",
	    nbytes);
    }

    spin_unlock(&kgpudev.gmpool_lock);
    return p;	
}
EXPORT_SYMBOL_GPL(kgpu_vmalloc);

void kgpu_vfree(void *p)
{
    unsigned long idx =
	((unsigned long)(p)
	 - (unsigned long)(kgpudev.gmpool.kva))/KGPU_BUF_UNIT_SIZE;
    unsigned int nunits;

    if (idx < 0 || idx >= kgpudev.gmpool.nunits) {
	kgpu_log(KGPU_LOG_ERROR,
		 "incorrect GPU memory pointer 0x%lX to free\n",
	    p);
	return;
    }

    nunits = kgpudev.gmpool.alloc_sz[idx];
    if (nunits == 0) {
	/*
	 * We allow such case because this allows users free memory
	 * from any field among in, out and data in request.
	 */
	return;
    }
    if (nunits > (kgpudev.gmpool.nunits - idx)) {
	kgpu_log(KGPU_LOG_ERROR, "incorrect GPU memory allocation info: "
		 "allocated %u units at unit index %u\n", nunits, idx);
	return;
    }

    spin_lock(&kgpudev.gmpool_lock);
    bitmap_clear(kgpudev.gmpool.bitmap, idx, nunits);
    kgpudev.gmpool.alloc_sz[idx] = 0;
    spin_unlock(&kgpudev.gmpool_lock);
}
EXPORT_SYMBOL_GPL(kgpu_vfree);

unsigned long kgpu_alloc_mmap_area(unsigned long size)
{
    unsigned int n = DIV_ROUND_UP(size, PAGE_SIZE);
    unsigned long p = 0;
    unsigned long idx;

    spin_lock(&kgpudev.vm_lock);

    idx = bitmap_find_next_zero_area(
	kgpudev.vm.bitmap,
	kgpudev.vm.npages,
	0, n, 0);
    if (idx < kgpudev.vm.npages) {
	bitmap_set(kgpudev.vm.bitmap, idx, n);
	p = kgpudev.vm.start + PAGE_SIZE*idx;
	kgpudev.vm.alloc_sz[idx] = n;
    } else {
	kgpu_log(KGPU_LOG_ERROR, "out of mmap area for mapping "
		 "ask for %u page %lu size, idx %lu\n",
		 n, size, idx);
    }

    spin_unlock(&kgpudev.vm_lock);

    return p;
}
EXPORT_SYMBOL_GPL(kgpu_alloc_mmap_area);

void kgpu_free_mmap_area(unsigned long start)
{
    unsigned long idx = (start - kgpudev.vm.start)>>PAGE_SHIFT;
    unsigned int  n;

    if (idx < 0 || idx >= kgpudev.vm.npages) {
	kgpu_log(KGPU_LOG_ERROR,
		 "incorrect GPU mmap pointer 0x%lX to free\n", start);
	return;
    }

    n = kgpudev.vm.alloc_sz[idx];
    if (n > (kgpudev.vm.npages - idx)) {
	kgpu_log(KGPU_LOG_ERROR,
		 "incorrect GPU mmap allocation info: "
		 "allocated %u pages at index %u\n", n, idx);
	return;
    }
    if (n > 0) {
	spin_lock(&kgpudev.vm_lock);
	bitmap_clear(kgpudev.vm.bitmap, idx, n);
	kgpudev.vm.alloc_sz[idx] = 0;
	spin_unlock(&kgpudev.vm_lock);
    }
}
EXPORT_SYMBOL_GPL(kgpu_free_mmap_area);


/*
 * This function has a lot of redundent code from kgpu_free_mmap_area,
 * we are going to remove that one.
 *
 * We can either unmap/zap the pte ranges to clear the page table entries,
 * or just leave them there until the next mapping, which *may*
 * overwrite page table entries without any problem.
 */
void kgpu_unmap_area(unsigned long addr)
{
    unsigned long idx = (addr-kgpudev.vm.start)>>PAGE_SHIFT;
    //return; // tmp workaround, this means we will consume the mem very soon...

    if (idx < 0 || idx >= kgpudev.vm.npages) {
	kgpu_log(KGPU_LOG_ERROR,
		 "incorrect GPU mmap pointer 0x%lX to unmap\n",
		 addr);
    } else {
	unsigned int n = kgpudev.vm.alloc_sz[idx];
	if (n > (kgpudev.vm.npages - idx)) {
	    kgpu_log(KGPU_LOG_ERROR,
		     "incorrect GPU mmap allocation info: "
		     "allocated %u pages at index %u\n", n, idx);
	    return;
	}
	//kgpu_log(KGPU_LOG_PRINT, "unmap %d pages from %p\n", n, addr);
	if (n > 0) {
	    int ret;
	    spin_lock(&kgpudev.vm_lock);
	    bitmap_clear(kgpudev.vm.bitmap, idx, n);
	    kgpudev.vm.alloc_sz[idx] = 0;

	    /*unmap_mapping_range(kgpudev.vm.vma->vm_file->f_mapping,
	      addr, n<<PAGE_SHIFT, 1);*/

	    kgpudev.vm.vma->vm_flags |= VM_PFNMAP;
	    ret = zap_vma_ptes(kgpudev.vm.vma, addr, n<<PAGE_SHIFT);
	    if (ret)
		kgpu_log(KGPU_LOG_ALERT,
			 "zap_vma_ptes returns %d\n", ret);
	    kgpudev.vm.vma->vm_flags &= ~VM_PFNMAP;
	    spin_unlock(&kgpudev.vm_lock);
	}
    }
}
EXPORT_SYMBOL_GPL(kgpu_unmap_area);

int kgpu_map_page(struct page *p, unsigned long addr)
{
    int ret = 0;
    down_write(&kgpudev.vm.vma->vm_mm->mmap_sem);

    ret = vm_insert_page(kgpudev.vm.vma, addr, p);
    if (unlikely(ret < 0)) {
	kgpu_log(KGPU_LOG_ERROR,
		 "can't remap pfn %lu, error %d ct %d\n",
		 page_to_pfn(p), ret, page_count(p));
    }

    up_write(&kgpudev.vm.vma->vm_mm->mmap_sem);
    return ret;
}
EXPORT_SYMBOL_GPL(kgpu_map_page);

static void* map_page_units(void *units, int n, int is_page)
{
    unsigned long addr;
    int i;
    int ret;

    addr = kgpu_alloc_mmap_area(n<<PAGE_SHIFT);
    if (!addr) {
	return NULL;
    }

    down_write(&kgpudev.vm.vma->vm_mm->mmap_sem);

    for (i=0; i<n; i++) {
	ret = vm_insert_page(
	    kgpudev.vm.vma,
	    addr+i*PAGE_SIZE,
	    is_page? ((struct page**)units)[i] : pfn_to_page(((unsigned long*)units)[i])
	    );
	
	if (unlikely(ret < 0)) {
	    up_write(&kgpudev.vm.vma->vm_mm->mmap_sem);
	    kgpu_log(KGPU_LOG_ERROR,
		     "can't remap pfn %lu, error code %d\n",
		     is_page ? page_to_pfn(((struct page**)units)[i]) : ((unsigned long*)units)[i],
		     ret);
	    return NULL;
	}
    }

    up_write(&kgpudev.vm.vma->vm_mm->mmap_sem);

    return (void*)addr;
    
}

void *kgpu_map_pages(struct page **pages, int n)
{
    return map_page_units(pages, n, 1);
}
EXPORT_SYMBOL_GPL(kgpu_map_pages);

void *kgpu_map_pfns(unsigned long *pfns, int n)
{
    return map_page_units(pfns, n, 0);
}
EXPORT_SYMBOL_GPL(kgpu_map_pfns);

/*
 * find request by id in the rtdreqs
 * offlist = 1: remove the request from the list
 * offlist = 0: keep the request in the list
 */
static struct _kgpu_request_item* find_request(int id, int offlist)
{
    struct _kgpu_request_item *pos, *n;

    spin_lock(&(kgpudev.rtdreqlock));
    
    list_for_each_entry_safe(pos, n, &(kgpudev.rtdreqs), list) {
	if (pos->r->id == id) {
	    if (offlist)
		list_del(&pos->list);
	    spin_unlock(&(kgpudev.rtdreqlock));
	    return pos;
	}
    }

    spin_unlock(&(kgpudev.rtdreqlock));

    return NULL;
}


int kgpu_open(struct inode *inode, struct file *filp)
{
    if (!atomic_dec_and_test(&kgpudev_av)) {
	atomic_inc(&kgpudev_av);
	return -EBUSY;
    }

    filp->private_data = &kgpudev;
    return 0;
}

int kgpu_release(struct inode *inode, struct file *file)
{
    atomic_set(&kgpudev_av, 1);
    return 0;
}

static void fill_ku_request(struct kgpu_ku_request *kureq,
			   struct kgpu_request *req)
{
    kureq->id = req->id;
    memcpy(kureq->service_name, req->service_name, KGPU_SERVICE_NAME_SIZE);

    if (ADDR_WITHIN(req->in, kgpudev.gmpool.kva,
		    kgpudev.gmpool.npages<<PAGE_SHIFT)) {
	kureq->in = (void*)ADDR_REBASE(kgpudev.gmpool.uva,
				       kgpudev.gmpool.kva,
				       req->in);
    } else {
	kureq->in = req->in;
    }

    if (ADDR_WITHIN(req->out, kgpudev.gmpool.kva,
		    kgpudev.gmpool.npages<<PAGE_SHIFT)) {
	kureq->out = (void*)ADDR_REBASE(kgpudev.gmpool.uva,
				       kgpudev.gmpool.kva,
				       req->out);
    } else {
	kureq->out = req->out;
    }

    if (ADDR_WITHIN(req->udata, kgpudev.gmpool.kva,
		    kgpudev.gmpool.npages<<PAGE_SHIFT)) {
	kureq->data = (void*)ADDR_REBASE(kgpudev.gmpool.uva,
				       kgpudev.gmpool.kva,
				       req->udata);
    } else {
	kureq->data = req->udata;
    }
    
    kureq->insize = req->insize;
    kureq->outsize = req->outsize;
    kureq->datasize = req->udatasize;
}

ssize_t kgpu_read(
    struct file *filp, char __user *buf, size_t c, loff_t *fpos)
{
    ssize_t ret = 0;
    struct list_head *r;
    struct _kgpu_request_item *item;

    spin_lock(&(kgpudev.reqlock));
    while (list_empty(&(kgpudev.reqs))) {
	spin_unlock(&(kgpudev.reqlock));

	if (filp->f_flags & O_NONBLOCK)
	    return -EAGAIN;

	if (wait_event_interruptible(
		kgpudev.reqq, (!list_empty(&(kgpudev.reqs)))))
	    return -ERESTARTSYS;
	spin_lock(&(kgpudev.reqlock));
    }

    r = kgpudev.reqs.next;
    list_del(r);
    item = list_entry(r, struct _kgpu_request_item, list);
    if (item) {
	struct kgpu_ku_request kureq;
	fill_ku_request(&kureq, item->r);
	
	memcpy(buf, &kureq, sizeof(struct kgpu_ku_request));
	ret = c;
    }

    spin_unlock(&(kgpudev.reqlock));

    if (ret > 0 && item) {
	spin_lock(&(kgpudev.rtdreqlock));

	INIT_LIST_HEAD(&item->list);
	list_add_tail(&item->list, &(kgpudev.rtdreqs));

	spin_unlock(&(kgpudev.rtdreqlock));
    }
    
    *fpos += ret;

    return ret;    
}

ssize_t kgpu_write(struct file *filp, const char __user *buf,
		   size_t count, loff_t *fpos)
{
    struct kgpu_ku_response kuresp;
    struct _kgpu_request_item *item;
    ssize_t ret = 0;
    size_t  realcount;
    
    if (count < sizeof(struct kgpu_ku_response))
	ret = -EINVAL; /* Too small. */
    else
    {
	realcount = sizeof(struct kgpu_ku_response);

	memcpy/*copy_from_user*/(&kuresp, buf, realcount);

	item = find_request(kuresp.id, 1);
	if (!item)
	{	    
	    ret = -EFAULT; /* no request found */
	} else {
	    item->r->errcode = kuresp.errcode;
	    if (unlikely(kuresp.errcode != 0)) {
		switch(kuresp.errcode) {
		case KGPU_NO_RESPONSE:
		    kgpu_log(KGPU_LOG_ALERT,
			"userspace helper doesn't give any response\n");
		    break;
		case KGPU_NO_SERVICE:
		    kgpu_log(KGPU_LOG_ALERT,
			     "no such service %s\n",
			     item->r->service_name);
		    break;
		case KGPU_TERMINATED:
		    kgpu_log(KGPU_LOG_ALERT,
			     "request is terminated\n"
			);
		    break;
		default:
		    kgpu_log(KGPU_LOG_ALERT,
			     "unknown error with code %d\n",
			     kuresp.id);
		    break;		    
		}
	    }

	    /*
	     * Different strategy should be applied here:
	     * #1 invoke the callback in the write syscall, like here.
	     * #2 add the resp into the resp-list in the write syscall
	     *    and return, a kernel thread will process the list
	     *    and invoke the callback.
	     *
	     * Currently, the first one is used because this can ensure
	     * the fast response. A kthread may have to sleep so that
	     * the response can't be processed ASAP.
	     */
	    item->r->callback(item->r);
	    ret = count;/*realcount;*/
	    *fpos += ret;
	    kmem_cache_free(kgpu_request_item_cache, item);
	}
    }

    return ret;
}

static int clear_gpu_mempool(void)
{
    struct _kgpu_mempool *gmp = &kgpudev.gmpool;

    spin_lock(&kgpudev.gmpool_lock);
    if (gmp->pages)
	kfree(gmp->pages);
    if (gmp->bitmap)
	kfree(gmp->bitmap);
    if (gmp->alloc_sz)
	kfree(gmp->alloc_sz);

    vunmap((void*)gmp->kva);
    spin_unlock(&kgpudev.gmpool_lock);
    return 0;
}

static int set_gpu_mempool(char __user *buf)
{
    struct kgpu_gpu_mem_info gb;
    struct _kgpu_mempool *gmp = &kgpudev.gmpool;
    int i;
    int err=0;

    spin_lock(&(kgpudev.gmpool_lock));
    
    copy_from_user(&gb, buf, sizeof(struct kgpu_gpu_mem_info));

    /* set up pages mem */
    gmp->uva = (unsigned long)(gb.uva);
    gmp->npages = gb.size/PAGE_SIZE;
    if (!gmp->pages) {
	gmp->pages = kmalloc(sizeof(struct page*)*gmp->npages, GFP_KERNEL);
	if (!gmp->pages) {
	    kgpu_log(KGPU_LOG_ERROR, "run out of memory for gmp pages\n");
	    err = -ENOMEM;
	    goto unlock_and_out;
	}
    }

    for (i=0; i<gmp->npages; i++)
	gmp->pages[i]= kgpu_v2page(
	    (unsigned long)(gb.uva) + i*PAGE_SIZE
	    );

    /* set up bitmap */
    gmp->nunits = gmp->npages/KGPU_BUF_NR_FRAMES_PER_UNIT;
    if (!gmp->bitmap) {
	gmp->bitmap = kmalloc(
	    BITS_TO_LONGS(gmp->nunits)*sizeof(long), GFP_KERNEL);
	if (!gmp->bitmap) {
	    kgpu_log(KGPU_LOG_ERROR, "run out of memory for gmp bitmap\n");
	    err = -ENOMEM;
	    goto unlock_and_out;
	}
    }    
    bitmap_zero(gmp->bitmap, gmp->nunits);

    /* set up allocated memory sizes */
    if (!gmp->alloc_sz) {
	gmp->alloc_sz = kmalloc(
	    gmp->nunits*sizeof(u32), GFP_KERNEL);
	if (!gmp->alloc_sz) {
	    kgpu_log(KGPU_LOG_ERROR,
		     "run out of memory for gmp alloc_sz\n");
	    err = -ENOMEM;
	    goto unlock_and_out;
	}
    }
    memset(gmp->alloc_sz, 0, gmp->nunits);

    /* set up kernel remapping */
    gmp->kva = (unsigned long)vmap(
	gmp->pages, gmp->npages, GFP_KERNEL, PAGE_KERNEL);
    if (!gmp->kva) {
	kgpu_log(KGPU_LOG_ERROR, "map pages into kernel failed\n");
	err = -EFAULT;
	goto unlock_and_out;
    }
    
unlock_and_out:
    spin_unlock(&(kgpudev.gmpool_lock));    

    return err;
}

static int dump_gpu_bufs(char __user *buf)
{
    /* TODO: dump gmpool's info to buf */
    return 0;
}

static int terminate_all_requests(void)
{
    /* TODO: stop receiving requests, set all reqeusts code to
     KGPU_TERMINATED and call their callbacks */
    kgpudev.state = KGPU_TERMINATED;
    return 0;
}

static long kgpu_ioctl(struct file *filp,
	       unsigned int cmd, unsigned long arg)
{
    int err = 0;
    
    if (_IOC_TYPE(cmd) != KGPU_IOC_MAGIC)
	return -ENOTTY;
    if (_IOC_NR(cmd) > KGPU_IOC_MAXNR) return -ENOTTY;

    if (_IOC_DIR(cmd) & _IOC_READ)
	err = !access_ok(VERIFY_WRITE, (void __user *)arg, _IOC_SIZE(cmd));
    else if (_IOC_DIR(cmd) & _IOC_WRITE)
	err = !access_ok(VERIFY_READ, (void __user *)arg, _IOC_SIZE(cmd));
    if (err) return -EFAULT;

    switch (cmd) {
	
    case KGPU_IOC_SET_GPU_BUFS:
	err = set_gpu_mempool((char*)arg);
	break;
	
    case KGPU_IOC_GET_GPU_BUFS:
	err = dump_gpu_bufs((char*)arg);
	break;

    case KGPU_IOC_SET_STOP:
	err = terminate_all_requests();
	break;

    default:
	err = -ENOTTY;
	break;
    }

    return err;
}

static unsigned int kgpu_poll(struct file *filp, poll_table *wait)
{
    unsigned int mask = 0;
    
    spin_lock(&(kgpudev.reqlock));
    
    poll_wait(filp, &(kgpudev.reqq), wait);

    if (!list_empty(&(kgpudev.reqs))) 
	mask |= POLLIN | POLLRDNORM;

    mask |= POLLOUT | POLLWRNORM;

    spin_unlock(&(kgpudev.reqlock));

    return mask;
}

static void kgpu_vm_open(struct vm_area_struct *vma)
{
    // just let it go
}

static void kgpu_vm_close(struct vm_area_struct *vma)
{
    // nothing we can do now
}

static int kgpu_vm_fault(struct vm_area_struct *vma,
			 struct vm_fault *vmf)
{
    /*static struct page *p = NULL;
    if (!p) {
	p = alloc_page(GFP_KERNEL);
	if (p) {
	    kgpu_log(KGPU_LOG_PRINT,
		     "first page fault, 0x%lX (0x%lX)\n",
		     TO_UL(vmf->virtual_address),
		     TO_UL(vma->vm_start));
	    vmf->page = p;
	    return 0;
	} else {
	    kgpu_log(KGPU_LOG_ERROR,
		     "first page fault, kgpu mmap no page ");
	}
	}*/
    /* should never call this */
    kgpu_log(KGPU_LOG_ERROR,
	     "kgpu mmap area being accessed without pre-mapping 0x%lX (0x%lX)\n",
	     (unsigned long)vmf->virtual_address,
	     (unsigned long)vma->vm_start);
    vmf->flags |= VM_FAULT_NOPAGE|VM_FAULT_ERROR;
    return VM_FAULT_SIGBUS;
}

static struct vm_operations_struct kgpu_vm_ops = {
    .open  = kgpu_vm_open,
    .close = kgpu_vm_close,
    .fault = kgpu_vm_fault,
};

static void set_vm(struct vm_area_struct *vma)
{
    kgpudev.vm.vma = vma;
    kgpudev.vm.start = vma->vm_start;
    kgpudev.vm.end = vma->vm_end;
    kgpudev.vm.npages = (vma->vm_end - vma->vm_start)>>PAGE_SHIFT;
    kgpudev.vm.alloc_sz = kmalloc(
	sizeof(u32)*kgpudev.vm.npages, GFP_KERNEL);
    kgpudev.vm.bitmap = kmalloc(
	BITS_TO_LONGS(kgpudev.vm.npages)*sizeof(long), GFP_KERNEL);
    if (!kgpudev.vm.alloc_sz || !kgpudev.vm.bitmap) {
	kgpu_log(KGPU_LOG_ERROR,
		 "out of memory for vm's bitmap and records\n");
	if (kgpudev.vm.alloc_sz) kfree(kgpudev.vm.alloc_sz);
	if (kgpudev.vm.bitmap) kfree(kgpudev.vm.bitmap);
	kgpudev.vm.alloc_sz = NULL;
	kgpudev.vm.bitmap = NULL;
    };
    bitmap_zero(kgpudev.vm.bitmap, kgpudev.vm.npages);
    memset(kgpudev.vm.alloc_sz, 0, sizeof(u32)*kgpudev.vm.npages);
}

static void clean_vm(void)
{
    if (kgpudev.vm.alloc_sz)
	kfree(kgpudev.vm.alloc_sz);
    if (kgpudev.vm.bitmap)
	kfree(kgpudev.vm.bitmap);
}

static int kgpu_mmap(struct file *filp, struct vm_area_struct *vma)
{
    if (vma->vm_end - vma->vm_start != KGPU_MMAP_SIZE) {
	kgpu_log(KGPU_LOG_ALERT,
		 "mmap size incorrect from 0x$lX to 0x%lX with "
		 "%lu bytes\n", vma->vm_start, vma->vm_end,
		 vma->vm_end-vma->vm_start);
	return -EINVAL;
    }
    vma->vm_ops = &kgpu_vm_ops;
    vma->vm_flags |= VM_RESERVED;
    set_vm(vma);
    return 0;
}

static struct file_operations kgpu_ops =  {
    .owner          = THIS_MODULE,
    .read           = kgpu_read,
    .write          = kgpu_write,
    .poll           = kgpu_poll,
    .unlocked_ioctl = kgpu_ioctl,
    .open           = kgpu_open,
    .release        = kgpu_release,
    .mmap           = kgpu_mmap,
};

static int kgpu_init(void)
{
    int result = 0;
    int devno;

    kgpudev.state = KGPU_OK;
    
    INIT_LIST_HEAD(&(kgpudev.reqs));
    INIT_LIST_HEAD(&(kgpudev.rtdreqs));
    
    spin_lock_init(&(kgpudev.reqlock));
    spin_lock_init(&(kgpudev.rtdreqlock));

    init_waitqueue_head(&(kgpudev.reqq));

    spin_lock_init(&(kgpudev.ridlock));
    spin_lock_init(&(kgpudev.gmpool_lock));
    spin_lock_init(&(kgpudev.vm_lock));

    memset(&kgpudev.vm, 0, sizeof(struct _kgpu_vma));
    
    kgpudev.rid_sequence = 0;

    kgpu_request_cache = kmem_cache_create(
	"kgpu_request_cache", sizeof(struct kgpu_request), 0,
	SLAB_HWCACHE_ALIGN, kgpu_request_constructor);
    if (!kgpu_request_cache) {
	kgpu_log(KGPU_LOG_ERROR, "can't create request cache\n");
	return -EFAULT;
    }
    
    kgpu_request_item_cache = kmem_cache_create(
	"kgpu_request_item_cache", sizeof(struct _kgpu_request_item), 0,
	SLAB_HWCACHE_ALIGN, kgpu_request_item_constructor);
    if (!kgpu_request_item_cache) {
	kgpu_log(KGPU_LOG_ERROR, "can't create request item cache\n");
	kmem_cache_destroy(kgpu_request_cache);
	return -EFAULT;
    }

    kgpu_sync_call_data_cache = kmem_cache_create(
	"kgpu_sync_call_data_cache", sizeof(struct _kgpu_sync_call_data), 0,
	SLAB_HWCACHE_ALIGN, kgpu_sync_call_data_constructor);
    if (!kgpu_sync_call_data_cache) {
	kgpu_log(KGPU_LOG_ERROR, "can't create sync call data cache\n");
	kmem_cache_destroy(kgpu_request_cache);
	kmem_cache_destroy(kgpu_request_item_cache);
	return -EFAULT;
    }

    
    /* initialize buffer info */
    memset(&kgpudev.gmpool, 0, sizeof(struct _kgpu_mempool));

    /* dev class */
    kgpudev.cls = class_create(THIS_MODULE, "KGPU_DEV_NAME");
    if (IS_ERR(kgpudev.cls)) {
	result = PTR_ERR(kgpudev.cls);
	kgpu_log(KGPU_LOG_ERROR, "can't create dev class for KGPU\n");
	return result;
    }

    /* alloc dev */	
    result = alloc_chrdev_region(&kgpudev.devno, 0, 1, KGPU_DEV_NAME);
    devno = MAJOR(kgpudev.devno);
    kgpudev.devno = MKDEV(devno, 0);

    if (result < 0) {
        kgpu_log(KGPU_LOG_ERROR, "can't get major\n");
    } else {
	struct device *device;
	memset(&kgpudev.cdev, 0, sizeof(struct cdev));
	cdev_init(&kgpudev.cdev, &kgpu_ops);
	kgpudev.cdev.owner = THIS_MODULE;
	kgpudev.cdev.ops = &kgpu_ops;
	result = cdev_add(&kgpudev.cdev, kgpudev.devno, 1);
	if (result) {
	    kgpu_log(KGPU_LOG_ERROR, "can't add device %d", result);
	}

	/* create /dev/kgpu */
	device = device_create(kgpudev.cls, NULL, kgpudev.devno, NULL,
			       KGPU_DEV_NAME);
	if (IS_ERR(device)) {
	    kgpu_log(KGPU_LOG_ERROR, "creating device failed\n");
	    result = PTR_ERR(device);
	}
    }

    return result;
}

static void kgpu_cleanup(void)
{
    kgpudev.state = KGPU_TERMINATED;

    device_destroy(kgpudev.cls, kgpudev.devno);
    cdev_del(&kgpudev.cdev);
    class_destroy(kgpudev.cls);

    unregister_chrdev_region(kgpudev.devno, 1);
    if (kgpu_request_cache)
	kmem_cache_destroy(kgpu_request_cache);
    if (kgpu_request_item_cache)
	kmem_cache_destroy(kgpu_request_item_cache);
    if (kgpu_sync_call_data_cache)
	kmem_cache_destroy(kgpu_sync_call_data_cache);

    clear_gpu_mempool();
    clean_vm();
}

static int __init mod_init(void)
{
    kgpu_log(KGPU_LOG_PRINT, "KGPU loaded\n");
    return kgpu_init();
}

static void __exit mod_exit(void)
{
    kgpu_cleanup();
    kgpu_log(KGPU_LOG_PRINT, "KGPU unloaded\n");
}

module_init(mod_init);
module_exit(mod_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Weibin Sun");
MODULE_DESCRIPTION("GPU computing frameworka and driver");
