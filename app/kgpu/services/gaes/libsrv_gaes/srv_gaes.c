/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../../kgpu/kgpu.h"
#include "../../../kgpu/gputils.h"
#include "../gaesu.h"

#define BPT_BYTES_PER_BLOCK 4096

struct kgpu_service gaes_ecb_enc_srv;
struct kgpu_service gaes_ecb_dec_srv;

struct gaes_ecb_data {
    u32 *d_key;
    u32 *h_key;
    int nrounds;
    int nr_dblks_per_tblk;
};

CUmodule module;

int gaes_ecb_compute_size_bpt(struct kgpu_service_request *sr)
{
    sr->block_x =
	sr->outsize>=BPT_BYTES_PER_BLOCK?
	BPT_BYTES_PER_BLOCK/16: sr->outsize/16;
    sr->grid_x =
	sr->outsize/BPT_BYTES_PER_BLOCK?
	sr->outsize/BPT_BYTES_PER_BLOCK:1;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int gaes_ecb_launch_bpt(struct kgpu_service_request *sr)
{
    struct crypto_aes_ctx *hctx = (struct crypto_aes_ctx*)sr->hdata;
    struct crypto_aes_ctx *dctx = (struct crypto_aes_ctx*)sr->ddata;
	int offset;
	unsigned long nrounds; /* is "int" in device code 64-bit? */
	CUdeviceptr rkptr, textptr;
    CUfunction func;
    CUresult res;
    
    if (sr->s == &gaes_ecb_dec_srv) {
        nrounds = hctx->key_length/4+6;
        rkptr = (CUdeviceptr) dctx->key_dec;
		textptr = (CUdeviceptr) sr->dout;

        res = cuModuleGetFunction(&func, module, "_Z15aes_decrypt_bptPjiPh");
        if (res != CUDA_SUCCESS) {
            printf("cuModuleGetFunction() failed\n");
            return 0;
        }

#if 0
        res = cuFuncSetBlockShape(func, sr->block_x, sr->block_y, 1);
        if (res != CUDA_SUCCESS) {
            printf("cuFuncSetBlockShape() failed\n");
            return 0;
        }

		offset = 0;
		cuParamSetv(func, offset, &rkptr, sizeof(rkptr));
		offset += sizeof(rkptr);
		cuParamSetv(func, offset, &nrounds, sizeof(nrounds));
		offset += sizeof(nrounds);
		cuParamSetv(func, offset, &textptr, sizeof(textptr));
		offset += sizeof(textptr);
        cuParamSetSize(func, offset);

        res = cuLaunchGrid(func, sr->grid_x, sr->grid_y);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchGrid failed: res = %u\n", res);
            return 0;
        }
#else
		void *param[] = {&rkptr, &nrounds, &textptr};
		res = cuLaunchKernel(func, sr->grid_x, sr->grid_y, 1, 
							 sr->block_x, sr->block_y, 1, 
							 0, 0, (void**)param, NULL);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchKernel failed: res = %u\n", res);
            return -1;
        }
#endif
    }
    else {
        nrounds = hctx->key_length/4+6;
        rkptr = (CUdeviceptr) dctx->key_enc;
		textptr = (CUdeviceptr) sr->dout;

        res = cuModuleGetFunction(&func, module, "_Z15aes_encrypt_bptPjiPh");
        if (res != CUDA_SUCCESS) {
            printf("cuModuleGetFunction() failed\n");
            return 0;
        }

#if 0
        res = cuFuncSetBlockShape(func, sr->block_x, sr->block_y, 1);
        if (res != CUDA_SUCCESS) {
            printf("cuFuncSetBlockShape() failed\n");
            return 0;
        }

		offset = 0;
		cuParamSetv(func, offset, &rkptr, sizeof(rkptr));
		offset += sizeof(rkptr);
		cuParamSetv(func, offset, &nrounds, sizeof(nrounds));
		offset += sizeof(nrounds);
		cuParamSetv(func, offset, &textptr, sizeof(textptr));
		offset += sizeof(textptr);
        cuParamSetSize(func, offset);
        res = cuLaunchGrid(func, sr->grid_x, sr->grid_y);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchGrid failed: res = %u\n", res);
            return 0;
        }
#else
		void *param[] = {&rkptr, &nrounds, &textptr};
		res = cuLaunchKernel(func, sr->grid_x, sr->grid_y, 1, 
							 sr->block_x, sr->block_y, 1, 
							 0, 0, (void**)param, NULL);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchKernel failed: res = %u\n", res);
            return -1;
        }
#endif
    }

    return 0;
}

int gaes_ecb_prepare(struct kgpu_service_request *sr)
{
    cuMemcpyHtoD( (CUdeviceptr)sr->din, sr->hin, sr->insize );
    return 0;
}

int gaes_ecb_post(struct kgpu_service_request *sr)
{
    cuMemcpyDtoH( sr->hout, (CUdeviceptr)sr->dout, sr->outsize );
    return 0;
}


/*
 * Naming convention of ciphers:
 * g{algorithm}_{mode}[-({enc}|{dev})]
 *
 * {}  : var value
 * []  : optional
 * (|) : or
 */
int init_service(void *lh, int (*reg_srv)(struct kgpu_service*, void*))
{
    CUresult res;
    int err;
    printf("[libsrv_gaes] Info: init gaes services\n");
    
    sprintf(gaes_ecb_enc_srv.name, "gaes_ecb-enc");
    gaes_ecb_enc_srv.sid = 0;
    gaes_ecb_enc_srv.compute_size = gaes_ecb_compute_size_bpt;
    gaes_ecb_enc_srv.launch = gaes_ecb_launch_bpt;
    gaes_ecb_enc_srv.prepare = gaes_ecb_prepare;
    gaes_ecb_enc_srv.post = gaes_ecb_post;
    
    sprintf(gaes_ecb_dec_srv.name, "gaes_ecb-dec");
    gaes_ecb_dec_srv.sid = 0;
    gaes_ecb_dec_srv.compute_size = gaes_ecb_compute_size_bpt;
    gaes_ecb_dec_srv.launch = gaes_ecb_launch_bpt;
    gaes_ecb_dec_srv.prepare = gaes_ecb_prepare;
    gaes_ecb_dec_srv.post = gaes_ecb_post;

    err = reg_srv(&gaes_ecb_enc_srv, lh);
    err |= reg_srv(&gaes_ecb_dec_srv, lh);
    if (err) {
    	fprintf(stderr,
		"[libsrv_gaes] Error: failed to register gaes services\n");
    } 

    res = cuModuleLoad(&module, "./gaes.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuModuleLoad() failed\n");
        return 0;
    }
    
    return err;
}

int finit_service(void *lh, int (*unreg_srv)(const char*))
{
    int err;
    printf("[libsrv_gaes] Info: finit gaes services\n");

    cuModuleUnload(module);
    
    err = unreg_srv(gaes_ecb_enc_srv.name);
    err |= unreg_srv(gaes_ecb_dec_srv.name);
    if (err) {
    	fprintf(stderr,
		"[libsrv_gaes] Error: failed to unregister gaes services\n");
    }
    
    return err;
}


