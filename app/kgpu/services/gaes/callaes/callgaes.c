/* -*- linux-c -*-
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/crypto.h>
#include <linux/scatterlist.h>
#include <linux/string.h>
#include <linux/gfp.h>
#include <linux/err.h>
#include <linux/jiffies.h>
#include <linux/timex.h>


char* AES_GENERIC = "ecb(aes-generic)";
char* AES_ASM = "ecb(aes-asm)";
char* AES = "ecb(aes)";

char* AES_GPU_GENERIC = "gaes_ecb(aes-generic)";
char* AES_GPU_ASM = "gaes_ecb(aes-asm)";
char* AES_GPU = "gaes_ecb(aes)";

char* CIPHER;

#define MAX_BLK_SIZE (32*1024*1024)
#define MIN_BLK_SIZE (4*1024)

#define TEST_TIMES 10

int test_gpu = 0;

static int skip_cpu=0;

module_param(skip_cpu, int, 0444);
MODULE_PARM_DESC(skip_cpu, "do not test CPU cipher, default 0 (No)");


#if 0

static void dump_page_content(u8 *p)
{
    int r,c;
    printk("dump page content:\n");
    for (r=0; r<16; r++) {
	for (c=0; c<32; c++)
	    printk("%02x ", p[r*32+c]);
	printk("\n");
    }
}

static void dump_hex(u8 *p, int sz)
{
    int i;
    printk("dump hex:\n");
    for (i=0; i<sz; i++)
	printk("%02x ", p[i]);
    printk("\n");
}

#endif /* test only */


void test_aes(void)
{
	struct crypto_blkcipher *tfm;
	struct blkcipher_desc desc;
	u32 bs;
	int i,j;
	u32 npages;
	
	struct scatterlist *src;
	struct scatterlist *dst;
	char *buf;
	char **ins, **outs;
	u8 *iv;
	
	unsigned int ret;
	
	u8 key[] = {0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x08, 0x0A, 0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x12,
	0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x08, 0x0A, 0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x12};
	
	npages = MAX_BLK_SIZE/PAGE_SIZE;

	iv = kmalloc(32, GFP_KERNEL);
	if (!iv) {
	    printk("taes Error: failed to alloc IV\n");
	    return;
	}
	
	src = kmalloc(npages*sizeof(struct scatterlist), __GFP_ZERO|GFP_KERNEL);
	if (!src) {
		printk("taes ERROR: failed to alloc src\n");		
		return;
	}
	dst = kmalloc(npages*sizeof(struct scatterlist), __GFP_ZERO|GFP_KERNEL);
	if (!dst) {
		printk("taes ERROR: failed to alloc dst\n");
		kfree(src);		
		return;
	}
	ins = kmalloc(npages*sizeof(char*), __GFP_ZERO|GFP_KERNEL);
	if (!ins) {
		printk("taes ERROR: failed to alloc ins\n");
		kfree(src);
		kfree(dst);
		return;
	}
	outs = kmalloc(npages*sizeof(char*), __GFP_ZERO|GFP_KERNEL);
	if (!outs) {
		printk("taes ERROR: failed to alloc outs\n");
		kfree(src);
		kfree(dst);
		kfree(ins);		
		return;
	}
	
	tfm = crypto_alloc_blkcipher(CIPHER, 0, 0);
	
	if (IS_ERR(tfm)) {
		printk("failed to load transform for %s: %ld\n", CIPHER,
			PTR_ERR(tfm));
		goto out;
	}
	desc.tfm = tfm;
	desc.flags = 0;
	desc.info = iv;

	if (test_gpu)
		ret = crypto_blkcipher_setkey(tfm, key, 16);
	else
		ret = crypto_blkcipher_setkey(tfm, key, 16);
	if (ret) {
		printk("setkey() failed flags=%x %lu\n",
				crypto_blkcipher_get_flags(tfm), sizeof(key));
	 	goto out;
	}
	
	sg_init_table(src, npages);
	for (i=0; i<npages; i++) {
		buf = (void *)__get_free_page(GFP_KERNEL);
		if (!buf) {
			printk("taes ERROR: alloc free page error\n");
			goto free_err_pages;
		}
		ins[i] = buf;
		strcpy(buf, "this is a plain text!");
		sg_set_buf(src+i, buf, PAGE_SIZE);
		buf = (void *)__get_free_page(GFP_KERNEL);
		if (!buf) {
			printk("taes ERROR: alloc free page error\n");
			goto free_err_pages;
		}
		outs[i] = buf;
		sg_set_buf(dst+i, buf, PAGE_SIZE);
	}
	
	for (bs = MAX_BLK_SIZE; bs >= MIN_BLK_SIZE; bs >>= 1) {
		struct timeval t0, t1;
		long int enc, dec;

		do_gettimeofday(&t0);
		for (j=0; j<TEST_TIMES; j++) {
			ret = crypto_blkcipher_encrypt_iv(&desc, dst, dst, bs);
			if (ret) {
				printk("taes ERROR: enc error\n");
				goto free_err_pages;
			}
		}
		do_gettimeofday(&t1);
		enc = 1000000*(t1.tv_sec-t0.tv_sec) + 
			((int)(t1.tv_usec) - (int)(t0.tv_usec));

		do_gettimeofday(&t0);
		for (j=0; j<TEST_TIMES; j++) {
			ret = crypto_blkcipher_decrypt_iv(&desc, src, src, bs);
			if (ret) {
				printk("taes ERROR: dec error\n");
				goto free_err_pages;
			}
		}
		do_gettimeofday(&t1);
		dec = 1000000*(t1.tv_sec-t0.tv_sec) + 
			((int)(t1.tv_usec) - (int)(t0.tv_usec));

		printk("%5s: Size %10u, enc BW: %6ld MB/s dec BW: %6ld MB/s\n",
		       test_gpu?"GAES":"CAES", bs, (bs*TEST_TIMES)/enc, (bs*TEST_TIMES)/dec);
	}
	
	
free_err_pages:
	for (i=0; i<npages && ins[i]; i++){		
		free_page((unsigned long)ins[i]);
	}
	for (i=0; i<npages && outs[i]; i++){
		free_page((unsigned long)outs[i]);
	}
out:
	kfree(src);
	kfree(dst);
	kfree(ins);
	kfree(outs);
	kfree(iv);
	crypto_free_blkcipher(tfm);	
}

static int __init taes_init(void)
{
	printk("test gaes loaded\n");
	CIPHER = AES_GPU_ASM;
	test_gpu = 1;
	test_aes();
	/* CIPHER = AES_GPU; */
	/* test_aes(); */
	/* CIPHER = AES_GPU; */
	/* test_aes(); */
	/* CIPHER = AES_ASM; */
	if (skip_cpu)
		return 0;
	test_gpu = 0;
	/* test_aes(); */
	/* CIPHER = AES; */
	/* test_aes(); */
	CIPHER = AES_GENERIC;
	test_aes();
	return 0;
}

static void __exit taes_exit(void)
{
	printk("test gaes unloaded\n");
}

module_init(taes_init);
module_exit(taes_exit);

MODULE_DESCRIPTION("Test CUDA AES-ECB");
MODULE_LICENSE("GPL");

