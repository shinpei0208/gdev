/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * For test purpose only, to be inclued by other src files.
 *
 */

#include <linux/timex.h>

static long test_gaes(size_t sz, int enc, const char *cipher)
{
    struct crypto_blkcipher *tfm;
    struct blkcipher_desc desc;
    int i;
    u32 npages, ret;
    
    struct scatterlist *src, *dst;
    char *buf, *mpool, **ins, **outs;
    u8 *iv;

    struct timeval t0, t1;
    long t = 0;
    
    u8 key[] = {0x00, 0x01, 0x02, 0x03, 0x05,
		0x06, 0x07, 0x08, 0x0A, 0x0B,
		0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x12};

    npages = DIV_ROUND_UP(sz, PAGE_SIZE);
    mpool = kmalloc(
	npages*2*(sizeof(struct scatterlist)+sizeof(char*))+32,
	__GFP_ZERO|GFP_KERNEL);
    if (!mpool) {
	g_log(KGPU_LOG_ERROR, "out of memory for test\n");
	return 0;
    }

    src = (struct scatterlist*)mpool;
    dst = (struct scatterlist*)mpool+npages*sizeof(struct scatterlist);
    ins = (char**)(mpool + 2*npages*sizeof(struct scatterlist));
    outs = (char**)(ins + npages);
    iv = ((char*)outs) + npages*sizeof(char*);

    tfm = crypto_alloc_blkcipher(cipher, 0, 0);
    if (IS_ERR(tfm)) {
	g_log(KGPU_LOG_ERROR, "failed to load transform for %s: %ld\n",
	      cipher,
	      PTR_ERR(tfm));
	goto out;
    }
    desc.tfm = tfm;
    desc.flags = 0;
    desc.info = iv;
    
    ret = crypto_blkcipher_setkey(tfm, key, sizeof(key));
    if (ret) {
	g_log(KGPU_LOG_ERROR, "setkey() failed flags=%x\n",
	       crypto_blkcipher_get_flags(tfm));
	goto out;
    }
    
    sg_init_table(src, npages);
    sg_init_table(dst, npages);
    for (i=0; i<npages; i++) {
	buf = (void *)__get_free_page(GFP_KERNEL);
	if (!buf) {
	    g_log(KGPU_LOG_ERROR, "alloc free page error\n");
	    goto free_err_pages;
	}
	ins[i] = buf;
	sg_set_buf(src+i, buf, PAGE_SIZE);

	buf = (void *)__get_free_page(GFP_KERNEL);
	if (!buf) {
	    g_log(KGPU_LOG_ERROR, "alloc free page error\n");
	    goto free_err_pages;
	}
	outs[i] = buf;
	sg_set_buf(dst+i, buf, PAGE_SIZE);
    }

    do_gettimeofday(&t0);
    if (enc)
	ret = crypto_blkcipher_encrypt_iv(&desc, dst, src, sz);
    else
	ret = crypto_blkcipher_decrypt_iv(&desc, dst, src, sz);
    do_gettimeofday(&t1);

    if (ret) {
	g_log(KGPU_LOG_ERROR, "dec/enc error\n");
	goto free_err_pages;
    }

    t = 1000000*(t1.tv_sec-t0.tv_sec) + 
	((int)(t1.tv_usec) - (int)(t0.tv_usec));

free_err_pages:
    for (i=0; i<npages && ins[i]; i++){		
	free_page((unsigned long)ins[i]);
    }
    for (i=0; i<npages && outs[i]; i++){
	free_page((unsigned long)outs[i]);
    }
out:
    kfree(mpool);
    crypto_free_blkcipher(tfm);

    return t;    
}
