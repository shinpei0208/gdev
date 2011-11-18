/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 * 
 * GPU accelerated AES-XTS cipher
 * The cipher and the algorithm are binded closely.
 *
 * This cipher is mostly derived from the crypto/xts.c in Linux kernel tree.
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

#include <crypto/b128ops.h>
#include <crypto/gf128mul.h>

#include "../../../kgpu/kgpu.h"
#include "../gaesk.h"

struct priv {
    struct crypto_cipher *child;
    struct crypto_cipher *tweak;
    struct crypto_xts_info info;
    struct crypto_aes_ctx aes_ctx;
};


/* customized log function */
#define g_log(level, ...) kgpu_do_log(level, "gaes_xts", ##__VA_ARGS__)
#define dbg(...) g_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)


struct gaes_xts_async_data {
    struct completion *c;             /* async-call completion */
    struct scatterlist *dst, *src;    /* crypt destination and source */
    struct blkcipher_desc *desc;      /* cipher descriptor */
    unsigned int sz;                  /* data size */
    void *expage;                     /* extra page for calling KGPU, if any */
    unsigned int offset;              /* offset within scatterlists */
};

static int zero_copy=0;

module_param(zero_copy, int, 0444);
MODULE_PARM_DESC(zero_copy, "use GPU mem zero-copy, default 0 (No)");

static int split_threshold=256;

module_param(split_threshold, int, 0444);
MODULE_PARM_DESC(split_threshold, "size(#pages) threshold for split, default 256");


static int setkey(struct crypto_tfm *parent, const u8 *key,
		  unsigned int keylen)
{
    struct priv *ctx = crypto_tfm_ctx(parent);
    struct crypto_cipher *child = ctx->tweak;
    u32 *flags = &parent->crt_flags;
    int err;

    /* key consists of keys of equal size concatenated, therefore
     * the length must be even */
    if (keylen % 2) {
	/* tell the user why there was an error */
	*flags |= CRYPTO_TFM_RES_BAD_KEY_LEN;
	return -EINVAL;
    }

    /* we need two cipher instances: one to compute the initial 'tweak'
     * by encrypting the IV (usually the 'plain' iv) and the other
     * one to encrypt and decrypt the data */

    /* tweak cipher, uses Key2 i.e. the second half of *key */
    crypto_cipher_clear_flags(child, CRYPTO_TFM_REQ_MASK);
    crypto_cipher_set_flags(child, crypto_tfm_get_flags(parent) &
			    CRYPTO_TFM_REQ_MASK);
    err = crypto_cipher_setkey(child, key + keylen/2, keylen/2);
    // for easy testing only, won't affect performance
    //err = crypto_cipher_setkey(child, key, keylen);
    if (err)
	return err;

    crypto_tfm_set_flags(parent, crypto_cipher_get_flags(child) &
			 CRYPTO_TFM_RES_MASK);

    child = ctx->child;

    /* data cipher, uses Key1 i.e. the first half of *key */
    crypto_cipher_clear_flags(child, CRYPTO_TFM_REQ_MASK);
    crypto_cipher_set_flags(child, crypto_tfm_get_flags(parent) &
			    CRYPTO_TFM_REQ_MASK);
    
    err = crypto_aes_expand_key(&ctx->aes_ctx,
				key, keylen/2);
    err = crypto_cipher_setkey(child, key, keylen/2);
    if (err)
	return err;

    cvt_endian_u32(ctx->aes_ctx.key_enc, AES_MAX_KEYLENGTH_U32);
    cvt_endian_u32(ctx->aes_ctx.key_dec, AES_MAX_KEYLENGTH_U32);

    memcpy(ctx->info.key_enc, ctx->aes_ctx.key_enc, AES_MAX_KEYLENGTH);
    memcpy(ctx->info.key_dec, ctx->aes_ctx.key_dec, AES_MAX_KEYLENGTH);
    ctx->info.key_length = ctx->aes_ctx.key_length;
    
    crypto_tfm_set_flags(parent, crypto_cipher_get_flags(child) &
			 CRYPTO_TFM_RES_MASK);

    return 0;
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
    struct gaes_xts_async_data *data = (struct gaes_xts_async_data*)
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

struct sinfo {
    be128 *t;
    struct crypto_tfm *tfm;
    void (*fn)(struct crypto_tfm *, u8 *, const u8 *);
};

static inline void xts_round(struct sinfo *s, void *dst, const void *src)
{
    be128_xor(dst, s->t, src);		/* PP <- T xor P */
    s->fn(s->tfm, dst, dst);		/* CC <- E(Key1,PP) */
    be128_xor(dst, dst, s->t);		/* C <- T xor CC */
}

static int cpu_crypt(struct blkcipher_desc *d,
		 struct blkcipher_walk *w, struct priv *ctx,
		 void (*tw)(struct crypto_tfm *, u8 *, const u8 *),
		 void (*fn)(struct crypto_tfm *, u8 *, const u8 *))
{
	int err;
	unsigned int avail;
	const int bs = crypto_cipher_blocksize(ctx->child);
	struct sinfo s = {
		.tfm = crypto_cipher_tfm(ctx->child),
		.fn = fn
	};
	u8 *wsrc;
	u8 *wdst;

	err = blkcipher_walk_virt(d, w);
	if (!w->nbytes)
		return err;

	s.t = (be128 *)w->iv;
	avail = w->nbytes;

	wsrc = w->src.virt.addr;
	wdst = w->dst.virt.addr;

	/* calculate first value of T */
	tw(crypto_cipher_tfm(ctx->tweak), w->iv, w->iv);

	goto first;

	for (;;) {
		do {
			gf128mul_x_ble(s.t, s.t);

first:
			xts_round(&s, wdst, wsrc);

			wsrc += bs;
			wdst += bs;
		} while ((avail -= bs) >= bs);

		err = blkcipher_walk_done(d, w, avail);
		if (!w->nbytes)
			break;

		avail = w->nbytes;

		wsrc = w->src.virt.addr;
		wdst = w->dst.virt.addr;
	}

	return err;
}

static void gpu_build_ivs(struct priv *ctx, u8 *iv,
    void (*tw)(struct crypto_tfm *, u8 *, const u8 *))
{
    int i;

    tw(crypto_cipher_tfm(ctx->tweak),
       (u8*)&(ctx->info.ivs[0]), iv);
    
    for (i=1; i<XTS_SECTOR_SIZE/AES_BLOCK_SIZE; i++) {
	gf128mul_x_ble(&(ctx->info.ivs[i]),
		       &(ctx->info.ivs[i-1]));
    }
}

static int gpu_crypt_nzc(struct blkcipher_desc *d,
			struct scatterlist *dst,
			struct scatterlist *src, struct priv *ctx,
			unsigned int sz,
			int enc, struct completion *c, unsigned int offset)
{
    int err=0;
    size_t rsz = roundup(sz, PAGE_SIZE);
    size_t nbytes;
    unsigned int cur;

    struct kgpu_request *req;
    char *buf;

    struct blkcipher_walk walk;
    
    buf = kgpu_vmalloc(rsz+sizeof(struct crypto_xts_info));
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
    req->insize = rsz+sizeof(struct crypto_xts_info);
    req->outsize = sz;
    req->udatasize = sizeof(struct crypto_xts_info);
    req->udata = buf+rsz;

    blkcipher_walk_init(&walk, dst, src, sz+offset);
    err = blkcipher_walk_virt(d, &walk);
    cur = 0;

    while ((nbytes = walk.nbytes)) {
	if (cur >= offset) {
	    u8 *wsrc = walk.src.virt.addr;
	    
	    memcpy(buf, wsrc, nbytes);
	    buf += nbytes;
	}

	cur += nbytes;	
	err = blkcipher_walk_done(d, &walk, 0);
	if (cur >= sz+offset)
	    break;
    }

    memcpy(req->udata, &(ctx->info), sizeof(struct crypto_xts_info));   
    strcpy(req->service_name, enc?"gaes_xts-enc":"gaes_xts-dec");

    if (c) {
	struct gaes_xts_async_data *adata =
	    kmalloc(sizeof(struct gaes_xts_async_data), GFP_KERNEL);
	if (!adata) {
	    g_log(KGPU_LOG_ERROR, "out of mem for async data\n");
	    // TODO: do something here
	} else {
	    req->callback = async_gpu_callback;
	    req->kdata = adata;

	    adata->c = c;
	    adata->dst = dst;
	    adata->src = src;
	    adata->desc = d;
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
	    __done_cryption(d, dst, src, sz, (char*)req->out, offset);
	}
	kgpu_vfree(req->in);
	kgpu_free_request(req); 
    }
    
    return err;
}

static int gpu_crypt_zc(struct blkcipher_desc *d,
			struct scatterlist *dst,
			struct scatterlist *src, struct priv *ctx,
			unsigned int sz,
			int enc, struct completion *c, unsigned int offset)
{
    int err;
    int i, n;
    unsigned int rsz = round_up(sz, PAGE_SIZE);
    int inplace = (sg_virt(dst) == sg_virt(src));

    struct kgpu_request *req;
    unsigned long addr;
    unsigned int pgoff;
    struct scatterlist *sg;

    char *data = (char*)__get_free_page(GFP_KERNEL);
    if (!data) {
	g_log(KGPU_LOG_ERROR, "out of memory for data\n");
	return -ENOMEM;
    }
    
    addr = kgpu_alloc_mmap_area(
	(inplace?rsz:2*rsz)+sizeof(struct crypto_xts_info));
    if (!addr) {
	free_page(TO_UL(data));
	g_log(KGPU_LOG_ERROR, "GPU buffer space is null for"
	      "size %u inplace %d\n", rsz, inplace);
	return -ENOMEM;
    }
    
    req  = kgpu_alloc_request();
    if (!req) {
	kgpu_free_mmap_area(addr);
	free_page(TO_UL(data));
	g_log(KGPU_LOG_ERROR, "can't allocate request\n");
	return -EFAULT;
    }

    req->in = (void*)addr;
    req->out = (void*)(inplace?addr:addr+rsz+PAGE_SIZE);
    req->insize = rsz+sizeof(struct crypto_xts_info);
    req->outsize = sz;
    req->udatasize = sizeof(struct crypto_xts_info);
    req->udata = (void*)(addr+rsz);

    memcpy(data, &(ctx->info), sizeof(struct crypto_xts_info));   
    strcpy(req->service_name, enc?"gaes_xts-enc":"gaes_xts-dec");

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
	struct gaes_xts_async_data *adata =
	    kmalloc(sizeof(struct gaes_xts_async_data), GFP_KERNEL);
	if (!adata) {
	    g_log(KGPU_LOG_ERROR, "out of mem for async data\n");
	    // TODO: do something here
	} else {
	    req->callback = async_gpu_callback;
	    req->kdata = adata;

	    adata->c = c;
	    adata->dst = dst;
	    adata->src = src;
	    adata->desc = d;
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

static int gpu_crypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes, int enc)
{
    struct priv *ctx = crypto_blkcipher_ctx(desc->tfm);
    gpu_build_ivs(ctx, (u8*)desc->info,
		  crypto_cipher_alg(ctx->tweak)->cia_encrypt);
    
    if ((nbytes>>PAGE_SHIFT)
	>= (split_threshold+(split_threshold>>1))) {
	unsigned int remainings = nbytes;
	int nparts = nbytes/(split_threshold<<(PAGE_SHIFT-1));
	struct completion *cs;
	int i;
	int ret = 0;

	if (nparts & 0x1)
	    nparts++;
	nparts >>= 1;

	cs = (struct completion*)kmalloc(sizeof(struct completion)*nparts,
					 GFP_KERNEL);
	if (cs) {
	    for(i=0; i<nparts && remainings > 0; i++) {
		init_completion(cs+i);
		if (zero_copy)
		    ret = gpu_crypt_zc(desc, dst, src, ctx,
				       (i==nparts-1)?remainings:
				       split_threshold<<PAGE_SHIFT,
				       enc, cs+i, i*(split_threshold<<PAGE_SHIFT));
		else
		    ret = gpu_crypt_nzc(desc, dst, src, ctx,
				    (i==nparts-1)?remainings:
				    split_threshold<<PAGE_SHIFT,
				    enc, cs+i, i*(split_threshold<<PAGE_SHIFT));
		
		if (ret < 0)
		    break;
		
		remainings -= (split_threshold<<PAGE_SHIFT);
	    }

	    for (i--; i>=0; i--)
		wait_for_completion_interruptible(cs+i);
	    kfree(cs);
	    return ret;
	}
    }
    
    return zero_copy?
	gpu_crypt_zc(desc, dst, src, ctx, nbytes, enc, NULL, 0):
	gpu_crypt_nzc(desc, dst, src, ctx, nbytes, enc, NULL, 0);
}


static int cpu_encrypt(struct blkcipher_desc *desc, struct scatterlist *dst,
		   struct scatterlist *src, unsigned int nbytes)
{
    struct priv *ctx = crypto_blkcipher_ctx(desc->tfm);
    struct blkcipher_walk w;

    blkcipher_walk_init(&w, dst, src, nbytes);
    return cpu_crypt(desc, &w, ctx, crypto_cipher_alg(ctx->tweak)->cia_encrypt,
		 crypto_cipher_alg(ctx->child)->cia_encrypt);
}

static int cpu_decrypt(struct blkcipher_desc *desc, struct scatterlist *dst,
		   struct scatterlist *src, unsigned int nbytes)
{
    struct priv *ctx = crypto_blkcipher_ctx(desc->tfm);
    struct blkcipher_walk w;

    blkcipher_walk_init(&w, dst, src, nbytes);
    return cpu_crypt(desc, &w, ctx, crypto_cipher_alg(ctx->tweak)->cia_encrypt,
		 crypto_cipher_alg(ctx->child)->cia_decrypt);
}

static int
gaes_xts_encrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{    
    if (/*nbytes%PAGE_SIZE != 0 ||*/ nbytes <= GAES_XTS_SIZE_THRESHOLD)
    	return cpu_encrypt(desc, dst, src, nbytes);
    return gpu_crypt(desc, dst, src, nbytes, 1);
}

static int
gaes_xts_decrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    if (/*nbytes%PAGE_SIZE != 0 ||*/ nbytes <= GAES_XTS_SIZE_THRESHOLD)
    	return cpu_decrypt(desc, dst, src, nbytes);
    return gpu_crypt(desc, dst, src, nbytes, 0);
}


static int init_tfm(struct crypto_tfm *tfm)
{
    struct crypto_cipher *cipher;
    struct crypto_instance *inst = (void *)tfm->__crt_alg;
    struct crypto_spawn *spawn = crypto_instance_ctx(inst);
    struct priv *ctx = crypto_tfm_ctx(tfm);
    u32 *flags = &tfm->crt_flags;

    cipher = crypto_spawn_cipher(spawn);
    if (IS_ERR(cipher))
	return PTR_ERR(cipher);

    if (crypto_cipher_blocksize(cipher) != 16) {
	*flags |= CRYPTO_TFM_RES_BAD_BLOCK_LEN;
	crypto_free_cipher(cipher);
	return -EINVAL;
    }

    ctx->child = cipher;

    cipher = crypto_spawn_cipher(spawn);
    if (IS_ERR(cipher)) {
	crypto_free_cipher(ctx->child);
	return PTR_ERR(cipher);
    }

    /* this check isn't really needed, leave it here just in case */
    if (crypto_cipher_blocksize(cipher) != 16) {
	crypto_free_cipher(cipher);
	crypto_free_cipher(ctx->child);
	*flags |= CRYPTO_TFM_RES_BAD_BLOCK_LEN;
	return -EINVAL;
    }

    ctx->tweak = cipher;

    return 0;
}

static void exit_tfm(struct crypto_tfm *tfm)
{
    struct priv *ctx = crypto_tfm_ctx(tfm);
    crypto_free_cipher(ctx->child);
    crypto_free_cipher(ctx->tweak);
}

static struct crypto_instance *alloc(struct rtattr **tb)
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

    inst = crypto_alloc_instance("gaes_xts", alg);
    if (IS_ERR(inst))
	goto out_put_alg;

    inst->alg.cra_flags = CRYPTO_ALG_TYPE_BLKCIPHER;
    inst->alg.cra_priority = alg->cra_priority;
    inst->alg.cra_blocksize = alg->cra_blocksize;

    if (alg->cra_alignmask < 7)
	inst->alg.cra_alignmask = 7;
    else
	inst->alg.cra_alignmask = alg->cra_alignmask;

    inst->alg.cra_type = &crypto_blkcipher_type;

    inst->alg.cra_blkcipher.ivsize = alg->cra_blocksize;
    inst->alg.cra_blkcipher.min_keysize =
	2 * alg->cra_cipher.cia_min_keysize;
    inst->alg.cra_blkcipher.max_keysize =
	2 * alg->cra_cipher.cia_max_keysize;

    inst->alg.cra_ctxsize = sizeof(struct priv);

    inst->alg.cra_init = init_tfm;
    inst->alg.cra_exit = exit_tfm;

    inst->alg.cra_blkcipher.setkey = setkey;
    inst->alg.cra_blkcipher.encrypt = gaes_xts_encrypt;
    inst->alg.cra_blkcipher.decrypt = gaes_xts_decrypt;

out_put_alg:
    crypto_mod_put(alg);
    return inst;
}

static void free(struct crypto_instance *inst)
{
    crypto_drop_spawn(crypto_instance_ctx(inst));
    kfree(inst);
}

static struct crypto_template crypto_tmpl = {
    .name = "gaes_xts",
    .alloc = alloc,
    .free = free,
    .module = THIS_MODULE,
};

static int __init crypto_module_init(void)
{
    if (!split_threshold) {
	g_log(KGPU_LOG_ERROR,
	      "incorrect split_threshold parameter %u\n",
	      split_threshold);
	split_threshold = 256;
    }
    g_log(KGPU_LOG_PRINT, "module load\n");
    return crypto_register_template(&crypto_tmpl);
}

static void __exit crypto_module_exit(void)
{
    g_log(KGPU_LOG_PRINT, "module unload\n");
    crypto_unregister_template(&crypto_tmpl);
}

module_init(crypto_module_init);
module_exit(crypto_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("GAES_XTS block cipher mode");
