/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 * 
 * GPU accelerated AES-ECB cipher
 * The cipher and the algorithm are binded closely.
 *
 * This cipher is mostly derived from the crypto/ecb.c in Linux kernel tree.
 *
 */
#include <crypto/algapi.h>
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <crypto/aes.h>
#include <linux/string.h>
#include <linux/completion.h>
#include "../../../kgpu/kgpu.h"
#include "../gaesk.h"

/* customized log function */
#define g_log(level, ...) kgpu_do_log(level, "gaes_ecb", ##__VA_ARGS__)
#define dbg(...) g_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)

struct crypto_gaes_ecb_ctx {
    struct crypto_cipher *child;
    struct crypto_aes_ctx aes_ctx;    
    u8 key[32];
};

struct gaes_ecb_async_data {
    struct completion *c;             /* async-call completion */
    struct scatterlist *dst, *src;    /* crypt destination and source */
    struct blkcipher_desc *desc;      /* cipher descriptor */
    unsigned int sz;                  /* data size */
    void *expage;                     /* extra page allocated before calling KGPU, if any */
    unsigned int offset;              /* offset within scatterlists */
};

static int zero_copy=0;
module_param(zero_copy, int, 0444);
MODULE_PARM_DESC(zero_copy, "use GPU mem zero-copy, default 0 (No)");

static int split_threshold=128;
module_param(split_threshold, int, 0444);
MODULE_PARM_DESC(split_threshold, "size(#pages) threshold for split, default 128");

static int max_splits=8;
module_param(max_splits, int, 0444);
MODULE_PARM_DESC(max_splits, "max number of sub-requests a big one be split, default 8");


static int
crypto_gaes_ecb_setkey(
    struct crypto_tfm *parent, const u8 *key,
    unsigned int keylen)
{
    struct crypto_gaes_ecb_ctx *ctx = crypto_tfm_ctx(parent);
    struct crypto_cipher *child = ctx->child;
    int err;

    crypto_cipher_clear_flags(child, CRYPTO_TFM_REQ_MASK);
    crypto_cipher_set_flags(child, crypto_tfm_get_flags(parent) &
			    CRYPTO_TFM_REQ_MASK);

    err = crypto_aes_expand_key(&ctx->aes_ctx,
				key, keylen);
    err = crypto_cipher_setkey(child, key, keylen);

    
    cvt_endian_u32(ctx->aes_ctx.key_enc, AES_MAX_KEYLENGTH_U32);
    cvt_endian_u32(ctx->aes_ctx.key_dec, AES_MAX_KEYLENGTH_U32);
    
    memcpy(ctx->key, key, keylen);
    
    crypto_tfm_set_flags(parent, crypto_cipher_get_flags(child) &
			 CRYPTO_TFM_RES_MASK);
    return err;
}

static void __done_cryption(struct blkcipher_desc *desc,
			    struct scatterlist *dst,
			    struct scatterlist *src,
			    unsigned int sz,
			    char *buf, unsigned int offset)
{
    struct blkcipher_walk walk;
    unsigned int nbytes, cur;
    
    blkcipher_walk_init(&walk, dst, src, sz+offset);
    blkcipher_walk_virt(desc, &walk);

    cur = 0;
 	
    while ((nbytes = walk.nbytes)) {
	if (cur >= offset) {
	    u8 *wdst = walk.dst.virt.addr;
	
	    memcpy(wdst, buf, nbytes);
	    buf += nbytes;
	}

	cur += nbytes;
	blkcipher_walk_done(desc, &walk, 0);
	if (cur >= sz+offset)
	    break;
    }
}

static int async_gpu_callback(struct kgpu_request *req)
{
    struct gaes_ecb_async_data *data = (struct gaes_ecb_async_data*)
	req->kdata;

    if (!zero_copy)
	__done_cryption(data->desc, data->dst, data->src, data->sz,
			(char*)req->out, data->offset);

    complete(data->c);

    if (zero_copy) {
	kgpu_unmap_area(TO_UL(req->in));
    } else
	kgpu_vfree(req->in);
    
    if (data->expage)
	free_page(TO_UL(data->expage));
    kgpu_free_request(req);

    kfree(data);
    return 0;
}

static int
crypto_gaes_ecb_crypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int sz,
    int enc, struct completion *c, unsigned int offset)
{
    int err=0;
    size_t rsz = roundup(sz, PAGE_SIZE);
    size_t nbytes;
    unsigned int cur;

    struct kgpu_request *req;
    char *buf;

    struct crypto_blkcipher *tfm    = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct blkcipher_walk walk;
    
    buf = kgpu_vmalloc(rsz+sizeof(struct crypto_aes_ctx));
    if (!buf) {
	g_log(KGPU_LOG_ERROR, "GPU buffer is null.\n");
	return -EFAULT;
    }

    req  = kgpu_alloc_request();
    if (!req) {
	kgpu_vfree(buf);
	g_log(KGPU_LOG_ERROR, "can't allocate request\n");
	return -EFAULT;
    }

    req->in = buf;
    req->out = buf;
    req->insize = rsz+sizeof(struct crypto_aes_ctx);
    req->outsize = sz;
    req->udatasize = sizeof(struct crypto_aes_ctx);
    req->udata = buf+rsz;

    blkcipher_walk_init(&walk, dst, src, sz+offset);
    err = blkcipher_walk_virt(desc, &walk);
    cur = 0;

    while ((nbytes = walk.nbytes)) {
	if (cur >= offset) {
	    u8 *wsrc = walk.src.virt.addr;
	    
	    memcpy(buf, wsrc, nbytes);
	    buf += nbytes;
	}

	cur += nbytes;	
	err = blkcipher_walk_done(desc, &walk, 0);
	if (cur >= sz+offset)
	    break;
    }

    memcpy(req->udata, &(ctx->aes_ctx), sizeof(struct crypto_aes_ctx));   
    strcpy(req->service_name, enc?"gaes_ecb-enc":"gaes_ecb-dec");

    if (c) {
	struct gaes_ecb_async_data *adata =
	    kmalloc(sizeof(struct gaes_ecb_async_data), GFP_KERNEL);
	if (!adata) {
	    g_log(KGPU_LOG_ERROR, "out of mem for async data\n");
	    // TODO: do something here
	} else {
	    req->callback = async_gpu_callback;
	    req->kdata = adata;

	    adata->c = c;
	    adata->dst = dst;
	    adata->src = src;
	    adata->desc = desc;
	    adata->sz = sz;
	    adata->expage = NULL;
	    adata->offset = offset;
	    kgpu_call_async(req);
	    return 0;
	}
    } else {
	if (kgpu_call_sync(req)) {
	    err = -EFAULT;
	    g_log(KGPU_LOG_ERROR, "callgpu error\n");
	} else {
	    __done_cryption(desc, dst, src, sz, (char*)req->out, offset);
	}
	kgpu_vfree(req->in);
	kgpu_free_request(req); 
    }
    
    return err;
}

static int
crypto_gaes_ecb_crypt_zc(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int sz,
    int enc, struct completion *c, unsigned int offset)
{
    int err=0;
    unsigned int rsz = roundup(sz, PAGE_SIZE);
    int inplace = (sg_virt(dst) == sg_virt(src));
    unsigned int pgoff;

    struct kgpu_request *req;
    unsigned long addr;
    int i, n;
    struct scatterlist *sg;

    struct crypto_blkcipher *tfm    = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);

    char *data = (char*)__get_free_page(GFP_KERNEL);
    if (!data) {
	g_log(KGPU_LOG_ERROR, "out of memory for data\n");
	return -ENOMEM;
    }
    
    addr = kgpu_alloc_mmap_area(
	(inplace?rsz:2*rsz)+sizeof(struct crypto_aes_ctx));
    if (!addr) {
	free_page(TO_UL(data));
	g_log(KGPU_LOG_ERROR, "GPU buffer space is null for"
	      "size %u inplace %d\n", rsz, inplace);
	return -ENOMEM;
    } /*else {
	g_log(KGPU_LOG_PRINT, "Got GPU buffer space %p for"
	      "size %u inplace %d\n", addr, rsz, inplace);
    } */

    req  = kgpu_alloc_request();
    if (!req) {
	kgpu_free_mmap_area(addr);
	free_page(TO_UL(data));
	g_log(KGPU_LOG_ERROR, "can't allocate request\n");
	return -EFAULT;
    }

    req->in = (void*)addr;
    req->out = (void*)(inplace?addr:addr+rsz+PAGE_SIZE);
    req->insize = rsz+sizeof(struct crypto_aes_ctx);
    req->outsize = sz;
    req->udatasize = sizeof(struct crypto_aes_ctx);
    req->udata = (void*)(addr+rsz);

    memcpy(data, &(ctx->aes_ctx), sizeof(struct crypto_aes_ctx));   
    strcpy(req->service_name, enc?"gaes_ecb-enc":"gaes_ecb-dec");

    pgoff = offset >> PAGE_SHIFT;
    n = pgoff + (rsz>>PAGE_SHIFT);
    for_each_sg(src, sg, n, i) {
	if (i >= pgoff) {	
	    if ((err = kgpu_map_page(sg_page(sg), addr)) < 0)
		goto get_out;
	    addr += PAGE_SIZE;
	}
    }

    if ((err = kgpu_map_page(virt_to_page(data), addr)) < 0)
	goto get_out;
    addr += PAGE_SIZE;

    if (!inplace)
	for_each_sg(dst, sg, n, i) {
	    if (i >= pgoff) {
		if ((err = kgpu_map_page(sg_page(sg), addr)) < 0)
		    goto get_out;
		addr += PAGE_SIZE;
	    }
	}

    if (c) {
	struct gaes_ecb_async_data *adata =
	    kmalloc(sizeof(struct gaes_ecb_async_data), GFP_KERNEL);
	if (!adata) {
	    g_log(KGPU_LOG_ERROR, "out of mem for async data\n");
	    // TODO: do something here
	} else {
	    req->callback = async_gpu_callback;
	    req->kdata = adata;

	    adata->c = c;
	    adata->dst = dst;
	    adata->src = src;
	    adata->desc = desc;
	    adata->sz = sz;
	    adata->expage = data;
	    adata->offset = offset;
	    kgpu_call_async(req);
	    return 0;
	}
    } else {
	if (kgpu_call_sync(req)) {
	    err = -EFAULT;
	    g_log(KGPU_LOG_ERROR, "callgpu error\n");
	}

	kgpu_unmap_area(TO_UL(req->in));
    get_out:
	kgpu_free_request(req);
	free_page(TO_UL(data));
    }

    return err;
}

static int crypto_ecb_gpu_crypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes, int enc)
{
    if ((nbytes>>PAGE_SHIFT)
	>= (split_threshold+(split_threshold>>1))) {
	unsigned int remainings = nbytes;
	int nparts = nbytes/(split_threshold<<(PAGE_SHIFT-1));
	struct completion *cs;
	int i;
	int ret = 0;
	unsigned int partsz = split_threshold<<PAGE_SHIFT;

	if (nparts & 0x1)
	    nparts++;
	nparts >>= 1;

	if (nparts > max_splits) {
	    nparts = max_splits;
	    partsz = nbytes/nparts;
	}

	cs = (struct completion*)kmalloc(sizeof(struct completion)*nparts,
					 GFP_KERNEL);
	if (cs) {
	    for(i=0; i<nparts && remainings > 0; i++) {
		init_completion(cs+i);
		if (zero_copy)
		    ret = crypto_gaes_ecb_crypt_zc(desc, dst, src,
						   (i==nparts-1)?remainings:
						   partsz,
						   enc, cs+i, i*partsz);
		else
		    ret = crypto_gaes_ecb_crypt(desc, dst, src,
						(i==nparts-1)?remainings:
						partsz,
						enc, cs+i, i*partsz);

		if (ret < 0)
		    break;
		
		remainings -= partsz;
	    }

	    for (i--; i>=0; i--)
		wait_for_completion_interruptible(cs+i);
	    kfree(cs);
	    return ret;
	}
    }
    
    return zero_copy?
	crypto_gaes_ecb_crypt_zc(desc, dst, src, nbytes, enc, NULL, 0):
	crypto_gaes_ecb_crypt(desc, dst, src, nbytes, enc, NULL, 0);
}


static int
crypto_ecb_crypt(
    struct blkcipher_desc *desc,
    struct blkcipher_walk *walk,
    struct crypto_cipher *tfm,
    void (*fn)(struct crypto_tfm *, u8 *, const u8 *))
{
    int bsize = crypto_cipher_blocksize(tfm);
    unsigned int nbytes;
    int err;

    err = blkcipher_walk_virt(desc, walk);

    while ((nbytes = walk->nbytes)) {
	u8 *wsrc = walk->src.virt.addr;
	u8 *wdst = walk->dst.virt.addr;

	do {
	    fn(crypto_cipher_tfm(tfm), wdst, wsrc);

	    wsrc += bsize;
	    wdst += bsize;
	} while ((nbytes -= bsize) >= bsize);

	err = blkcipher_walk_done(desc, walk, nbytes);
    }

    return err;
}

static int
crypto_ecb_encrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    struct blkcipher_walk walk;
    struct crypto_blkcipher *tfm = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct crypto_cipher *child = ctx->child;

    blkcipher_walk_init(&walk, dst, src, nbytes);
    return crypto_ecb_crypt(desc, &walk, child,
			    crypto_cipher_alg(child)->cia_encrypt);
}

static int
crypto_ecb_decrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    struct blkcipher_walk walk;
    struct crypto_blkcipher *tfm = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct crypto_cipher *child = ctx->child;

    blkcipher_walk_init(&walk, dst, src, nbytes);
    return crypto_ecb_crypt(desc, &walk, child,
			    crypto_cipher_alg(child)->cia_decrypt);
}

static int
crypto_gaes_ecb_encrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{    
    if (/*nbytes%PAGE_SIZE != 0 ||*/ nbytes <= GAES_ECB_SIZE_THRESHOLD)
    	return crypto_ecb_encrypt(desc, dst, src, nbytes);
    return crypto_ecb_gpu_crypt(desc, dst, src, nbytes, 1);
}

static int
crypto_gaes_ecb_decrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    if (/*nbytes%PAGE_SIZE != 0 ||*/ nbytes <= GAES_ECB_SIZE_THRESHOLD)
    	return crypto_ecb_decrypt(desc, dst, src, nbytes);
    return crypto_ecb_gpu_crypt(desc, dst, src, nbytes, 0);
}

static int crypto_gaes_ecb_init_tfm(struct crypto_tfm *tfm)
{
    struct crypto_instance *inst = (void *)tfm->__crt_alg;
    struct crypto_spawn *spawn = crypto_instance_ctx(inst);
    struct crypto_gaes_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
    struct crypto_cipher *cipher;

    cipher = crypto_spawn_cipher(spawn);
    if (IS_ERR(cipher))
	return PTR_ERR(cipher);

    ctx->child = cipher;
    return 0;
}

static void crypto_gaes_ecb_exit_tfm(struct crypto_tfm *tfm)
{
    struct crypto_gaes_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
    crypto_free_cipher(ctx->child);
}

static struct crypto_instance *crypto_gaes_ecb_alloc(struct rtattr **tb)
{
    struct crypto_instance *inst;
    struct crypto_alg *alg;
    int err;

    err = crypto_check_attr_type(tb, CRYPTO_ALG_TYPE_BLKCIPHER);
    if (err)
	return ERR_PTR(err);

    alg = crypto_get_attr_alg(tb, CRYPTO_ALG_TYPE_CIPHER,
			      CRYPTO_ALG_TYPE_MASK);
    if (IS_ERR(alg))
	return ERR_CAST(alg);

    inst = crypto_alloc_instance("gaes_ecb", alg);
    if (IS_ERR(inst)) {
	g_log(KGPU_LOG_ERROR, "cannot alloc crypto instance\n");
	goto out_put_alg;
    }

    inst->alg.cra_flags = CRYPTO_ALG_TYPE_BLKCIPHER;
    inst->alg.cra_priority = alg->cra_priority;
    inst->alg.cra_blocksize = alg->cra_blocksize;
    inst->alg.cra_alignmask = alg->cra_alignmask;
    inst->alg.cra_type = &crypto_blkcipher_type;

    inst->alg.cra_blkcipher.min_keysize = alg->cra_cipher.cia_min_keysize;
    inst->alg.cra_blkcipher.max_keysize = alg->cra_cipher.cia_max_keysize;

    inst->alg.cra_ctxsize = sizeof(struct crypto_gaes_ecb_ctx);

    inst->alg.cra_init = crypto_gaes_ecb_init_tfm;
    inst->alg.cra_exit = crypto_gaes_ecb_exit_tfm;

    inst->alg.cra_blkcipher.setkey = crypto_gaes_ecb_setkey;
    inst->alg.cra_blkcipher.encrypt = crypto_gaes_ecb_encrypt;
    inst->alg.cra_blkcipher.decrypt = crypto_gaes_ecb_decrypt;

out_put_alg:
    crypto_mod_put(alg);
    return inst;
}

static void crypto_gaes_ecb_free(struct crypto_instance *inst)
{
    crypto_drop_spawn(crypto_instance_ctx(inst));
    kfree(inst);
}

static struct crypto_template crypto_gaes_ecb_tmpl = {
    .name = "gaes_ecb",
    .alloc = crypto_gaes_ecb_alloc,
    .free = crypto_gaes_ecb_free,
    .module = THIS_MODULE,
};

#include "../gaes_test.c"

long test_gaes_ecb(size_t sz, int enc)
{
    return test_gaes(sz, enc, "gaes_ecb(aes)");
}
EXPORT_SYMBOL_GPL(test_gaes_ecb);

static int __init crypto_gaes_ecb_module_init(void)
{
    if (!split_threshold) {
	g_log(KGPU_LOG_ERROR,
	      "incorrect split_threshold parameter %u\n",
	      split_threshold);
	split_threshold = 1;
    }
    g_log(KGPU_LOG_PRINT, "module load\n");
    return crypto_register_template(&crypto_gaes_ecb_tmpl);
}

static void __exit crypto_gaes_ecb_module_exit(void)
{
    g_log(KGPU_LOG_PRINT, "module unload\n");
    crypto_unregister_template(&crypto_gaes_ecb_tmpl);
}

module_init(crypto_gaes_ecb_module_init);
module_exit(crypto_gaes_ecb_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gaes_ecb block cipher algorithm");
