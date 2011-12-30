/*
 * Copyright (c) 2011 Shinpei Kato
 *
 * University of California at Santa Cruz
 * Systems Research Lab
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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
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
#include "cuda.h"
#include "dirpath.h"

#define ECB_SPLIT_LAUNCH

#define ECB_GPU_ENC 0
#define ECB_GPU_DEC 1
#define ECB_GPU_SIZE_THRESHOLD (PAGE_SIZE-1)

struct crypto_ecb_gpu_ctx {
	struct crypto_cipher *child;
	struct crypto_aes_ctx aes_ctx;	
	u8 key[32];
};

#ifdef ECB_SPLIT_LAUNCH
static int split_threshold = 128;
module_param(split_threshold, int, 0444);
MODULE_PARM_DESC(split_threshold, 
				 "size(#pages) threshold for split, default 128");

static int max_splits = 8;
module_param(max_splits, int, 0444);
MODULE_PARM_DESC(max_splits, 
				 "max number of sub-requests a big one be split, default 8");
#endif

CUcontext ctx;
CUmodule module;

static void cvt_endian_u32(u32* buf, int n)
{
	u8* b = (u8*)buf;
	int nb = n*4;
	
	u8 t;
	int i;
	
	for (i=0; i<nb; i+=4, b+=4) {
		t = b[0];
		b[0] = b[3];
		b[3] = t;
		
		t = b[1];
		b[1] = b[2];
		b[2] = t;
	}
}

CUresult launch_bpt
(void *hdata, void *ddata, CUdeviceptr dout, int type, uint32_t size)
{
    struct crypto_aes_ctx *hctx = (struct crypto_aes_ctx*)hdata;
    struct crypto_aes_ctx *dctx = (struct crypto_aes_ctx*)ddata;
	int nr_threads = size / 16; /* specific to our aes_gpu.cu */
	int block_x = 32;
	int block_y = 1;
	int grid_x = nr_threads / block_x; /* should be < 1024 */
	int grid_y = 1;
	int offset;
	unsigned long nrounds;
    CUresult res;
    CUfunction func;
	CUdeviceptr rkptr, textptr;

    if (type == ECB_GPU_DEC) {
        nrounds = hctx->key_length/4+6;
        rkptr = (CUdeviceptr) dctx->key_dec;
		textptr = dout;

        res = cuModuleGetFunction(&func, module, "_Z15aes_decrypt_bptPjiPh");
        if (res != CUDA_SUCCESS) {
            printk("[ecb_gpu] cuModuleGetFunction() failed\n");
            return res;
        }
#if 0
        res = cuFuncSetBlockShape(func, block_x, block_y, 1);
        if (res != CUDA_SUCCESS) {
            printk("[ecb_gpu] cuFuncSetBlockShape() failed\n");
            return res;
        }

		offset = 0;
		cuParamSetv(func, offset, &rkptr, sizeof(rkptr));
		offset += sizeof(rkptr);
		cuParamSetv(func, offset, &nrounds, sizeof(nrounds));
		offset += sizeof(nrounds);
		cuParamSetv(func, offset, &textptr, sizeof(textptr));
		offset += sizeof(textptr);
		cuParamSetSize(func, offset);

        res = cuLaunchGrid(func, grid_x, grid_y);
        if (res != CUDA_SUCCESS) {
            printk("[ecb_gpu] cuLaunchGrid failed: res = %u\n", res);
            return res;
        }
#else
		void *param[] = {&rkptr, &nrounds, &textptr};
		res = cuLaunchKernel(func, sr->grid_x, sr->grid_y, 1, 
							 sr->block_x, sr->block_y, 1, 
							 0, 0, (void**)param, NULL);
        if (res != CUDA_SUCCESS) {
            printf("[ecb_gpu] cuLaunchKernel failed: res = %u\n", res);
            return -1;
        }
#endif
    }
    else {
        nrounds = hctx->key_length/4+6;
        rkptr = (CUdeviceptr) dctx->key_enc;
		textptr = dout;

        res = cuModuleGetFunction(&func, module, "_Z15aes_encrypt_bptPjiPh");
        if (res != CUDA_SUCCESS) {
            printk("[ecb_gpu] cuModuleGetFunction() failed\n");
            return res;
        }
#if 0
        res = cuFuncSetBlockShape(func, block_x, block_y, 1);
        if (res != CUDA_SUCCESS) {
            printk("[ecb_gpu] cuFuncSetBlockShape() failed\n");
            return res;
        }

		offset = 0;
		cuParamSetv(func, offset, &rkptr, sizeof(rkptr));
		offset += sizeof(rkptr);
		cuParamSetv(func, offset, &nrounds, sizeof(nrounds));
		offset += sizeof(nrounds);
		cuParamSetv(func, offset, &textptr, sizeof(textptr));
		offset += sizeof(textptr);
		cuParamSetSize(func, offset);

        res = cuLaunchGrid(func, grid_x, grid_y);
        if (res != CUDA_SUCCESS) {
            printk("[ecb_gpu] cuLaunchGrid failed: res = %u\n", res);
            return res;
        }
#else
		void *param[] = {&rkptr, &nrounds, &textptr};
		res = cuLaunchKernel(func, sr->grid_x, sr->grid_y, 1, 
							 sr->block_x, sr->block_y, 1, 
							 0, 0, (void**)param, NULL);
        if (res != CUDA_SUCCESS) {
            printf("[ecb_gpu] cuLaunchKernel failed: res = %u\n", res);
            return -1;
        }
#endif
    }

    return CUDA_SUCCESS;
}

static int crypto_ecb_gpu_setkey
(struct crypto_tfm *parent, const u8 *key, unsigned int keylen)
{
	struct crypto_ecb_gpu_ctx *ctx = crypto_tfm_ctx(parent);
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

static void __done_cryption
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int sz, char *buf, unsigned int offset)
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

static int ecb_gpu_crypt
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int sz, int enc, unsigned int offset)
{
	int err = 0;
	size_t rsz = roundup(sz, PAGE_SIZE);
	size_t nbytes;
	unsigned int cur;

	int page_locked;
	char *buf, *chunk;
	CUdeviceptr devptr;
	CUresult res;

	struct crypto_blkcipher *tfm	= desc->tfm;
	struct crypto_ecb_gpu_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct blkcipher_walk walk;

	struct crypto_aes_ctx *hctx, *dctx;
	
	res = cuMemAllocHost((void **)&buf, rsz + sizeof(struct crypto_aes_ctx));
	if (res != CUDA_SUCCESS) {
		page_locked = false;
		buf = vmalloc(rsz + sizeof(struct crypto_aes_ctx));
		if (!buf) {
			printk("[ecb_gpu] Failed to allocate host memory.\n");
			goto fail_buf;
		}
	}
	else {
		page_locked = true;
	}

	res = cuMemAlloc(&devptr, rsz + sizeof(struct crypto_aes_ctx));
	if (res != CUDA_SUCCESS) {
		printk("[ecb_gpu] Failed to allocate device memory.\n");
		goto fail_devptr;
	}

	blkcipher_walk_init(&walk, dst, src, sz+offset);
	err = blkcipher_walk_virt(desc, &walk);
	cur = 0;
	chunk = buf;
	
	while ((nbytes = walk.nbytes)) {
		if (cur >= offset) {
			u8 *wsrc = walk.src.virt.addr;
			
			memcpy(chunk, wsrc, nbytes);
			chunk += nbytes;
		}
		
		cur += nbytes;	
		err = blkcipher_walk_done(desc, &walk, 0);
		if (cur >= sz+offset)
			break;
	}

	hctx = &(ctx->aes_ctx);
    dctx = (struct crypto_aes_ctx*)(devptr + rsz);

	memcpy(buf + rsz, hctx, sizeof(struct crypto_aes_ctx));   
	res = cuMemcpyHtoD(devptr, buf, rsz + sizeof(struct crypto_aes_ctx));
	if (res != CUDA_SUCCESS)
		goto fail_memcpy_h2d;

	res = launch_bpt(hctx, dctx, devptr, enc ? ECB_GPU_ENC : ECB_GPU_DEC, rsz);
	if (res != CUDA_SUCCESS)
		goto fail_launch;

	res = cuMemcpyDtoH(buf, devptr, rsz);
	if (res != CUDA_SUCCESS)
		goto fail_memcpy_d2h;
	
	__done_cryption(desc, dst, src, sz, buf, offset);

	cuMemFree(devptr);
	if (page_locked)
		cuMemFreeHost(buf);
	else
		vfree(buf);

	return err;

fail_memcpy_d2h:
fail_launch:
fail_memcpy_h2d:
	cuMemFree(devptr);
fail_devptr:
	if (page_locked)
		cuMemFreeHost(buf);
	else
		vfree(buf);
fail_buf:
	return -EFAULT;
}

static int crypto_ecb_gpu_crypt
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int nbytes, int enc)
{
#ifdef ECB_SPLIT_LAUNCH
	if ((nbytes >> PAGE_SHIFT) >= (split_threshold + (split_threshold >> 1))) {
		unsigned int remainings = nbytes;
		int nparts = nbytes/(split_threshold << (PAGE_SHIFT - 1));
		int i;
		int ret = 0;
		unsigned int partsz = split_threshold << PAGE_SHIFT;

		if (nparts & 0x1)
			nparts++;
		nparts >>= 1;

		if (nparts > max_splits) {
			nparts = max_splits;
			partsz = nbytes/nparts;
		}

		for (i = 0; i < nparts && remainings > 0; i++) {
			ret = ecb_gpu_crypt(desc, dst, src,
								(i == nparts - 1) ? remainings : partsz,
								enc, i*partsz);
			if (ret < 0)
				break;
			
			remainings -= partsz;
		}
		
		return ret;
	}
#endif
	return ecb_gpu_crypt(desc, dst, src, nbytes, enc, 0);
}


static int crypto_ecb_crypt
(struct blkcipher_desc *desc, struct blkcipher_walk *walk, 
 struct crypto_cipher *tfm, void (*fn)(struct crypto_tfm *, u8 *, const u8 *))
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

static int crypto_ecb_encrypt
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int nbytes)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_ecb_gpu_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct crypto_cipher *child = ctx->child;

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_crypt(desc, &walk, child,
							crypto_cipher_alg(child)->cia_encrypt);
}

static int crypto_ecb_decrypt
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int nbytes)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_ecb_gpu_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct crypto_cipher *child = ctx->child;

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_crypt(desc, &walk, child,
							crypto_cipher_alg(child)->cia_decrypt);
}

static int crypto_ecb_gpu_encrypt
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int nbytes)
{	
	/* sometimes it's faster to compute on the CPU. */
	if (nbytes <= ECB_GPU_SIZE_THRESHOLD)
		return crypto_ecb_encrypt(desc, dst, src, nbytes);
	return crypto_ecb_gpu_crypt(desc, dst, src, nbytes, 1);
}

static int crypto_ecb_gpu_decrypt
(struct blkcipher_desc *desc, struct scatterlist *dst, struct scatterlist *src,
 unsigned int nbytes)
{
	/* sometimes it's faster to compute on the CPU. */
	if (nbytes <= ECB_GPU_SIZE_THRESHOLD)
		return crypto_ecb_decrypt(desc, dst, src, nbytes);
	return crypto_ecb_gpu_crypt(desc, dst, src, nbytes, 0);
}

static int crypto_ecb_gpu_init_tfm(struct crypto_tfm *tfm)
{
	struct crypto_instance *inst = (void *)tfm->__crt_alg;
	struct crypto_spawn *spawn = crypto_instance_ctx(inst);
	struct crypto_ecb_gpu_ctx *ctx = crypto_tfm_ctx(tfm);
	struct crypto_cipher *cipher;

	cipher = crypto_spawn_cipher(spawn);
	if (IS_ERR(cipher))
		return PTR_ERR(cipher);

	ctx->child = cipher;
	return 0;
}

static void crypto_ecb_gpu_exit_tfm(struct crypto_tfm *tfm)
{
	struct crypto_ecb_gpu_ctx *ctx = crypto_tfm_ctx(tfm);
	crypto_free_cipher(ctx->child);
}

static struct crypto_instance *crypto_ecb_gpu_alloc(struct rtattr **tb)
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

	inst = crypto_alloc_instance("ecb_gpu", alg);
	if (IS_ERR(inst)) {
		printk("[ecb_gpu] cannot alloc crypto instance\n");
		goto out_put_alg;
	}

	inst->alg.cra_flags = CRYPTO_ALG_TYPE_BLKCIPHER;
	inst->alg.cra_priority = alg->cra_priority;
	inst->alg.cra_blocksize = alg->cra_blocksize;
	inst->alg.cra_alignmask = alg->cra_alignmask;
	inst->alg.cra_type = &crypto_blkcipher_type;

	inst->alg.cra_blkcipher.min_keysize = alg->cra_cipher.cia_min_keysize;
	inst->alg.cra_blkcipher.max_keysize = alg->cra_cipher.cia_max_keysize;

	inst->alg.cra_ctxsize = sizeof(struct crypto_ecb_gpu_ctx);

	inst->alg.cra_init = crypto_ecb_gpu_init_tfm;
	inst->alg.cra_exit = crypto_ecb_gpu_exit_tfm;

	inst->alg.cra_blkcipher.setkey = crypto_ecb_gpu_setkey;
	inst->alg.cra_blkcipher.encrypt = crypto_ecb_gpu_encrypt;
	inst->alg.cra_blkcipher.decrypt = crypto_ecb_gpu_decrypt;

out_put_alg:
	crypto_mod_put(alg);
	return inst;
}

static void crypto_ecb_gpu_free(struct crypto_instance *inst)
{
	crypto_drop_spawn(crypto_instance_ctx(inst));
	kfree(inst);
}

static struct crypto_template crypto_ecb_gpu_tmpl = {
	.name = "ecb_gpu",
	.alloc = crypto_ecb_gpu_alloc,
	.free = crypto_ecb_gpu_free,
	.module = THIS_MODULE,
};

static int __init crypto_ecb_gpu_module_init(void)
{
	CUresult res;
    CUdevice dev;
	char fname[256];

#ifdef ECB_SPLIT_LAUNCH
	if (!split_threshold) {
		split_threshold = 1;
	}
#endif

    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        printk("[ecb_gpu] cuInit failed: res = %u\n", res);
        return 0;
    }

    res = cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        printk("[ecb_gpu] cuDeviceGet failed: res = %u\n", res);
        return 0;
    }

    res = cuCtxCreate(&ctx, 0, dev);
    if (res != CUDA_SUCCESS) {
        printk("[ecb_gpu] cuCtxCreate failed: res = %u\n", res);
        return 0;
    }
    
	sprintf(fname, "%s/aes_gpu.cubin", DIRPATH);
    res = cuModuleLoad(&module, fname);
    if (res != CUDA_SUCCESS) {
        printk("[ecb_gpu] cuModuleLoad() failed\n");
        return 0;
    }

	return crypto_register_template(&crypto_ecb_gpu_tmpl);
}

static void __exit crypto_ecb_gpu_module_exit(void)
{
	crypto_unregister_template(&crypto_ecb_gpu_tmpl);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);
}

module_init(crypto_ecb_gpu_module_init);
module_exit(crypto_ecb_gpu_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("GPU-accelerated block cipher algorithm");
