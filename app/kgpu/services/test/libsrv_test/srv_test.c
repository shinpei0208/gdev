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

CUmodule module;

int test_compute_size(struct kgpu_service_request *sr)
{
    sr->block_x = 32;
    sr->grid_x = sr->insize/256;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int test_launch(struct kgpu_service_request *sr)
{
    CUresult res;
    CUfunction func;

    res = cuModuleGetFunction(&func, module, "_Z10inc_kernelPiS_");
    if (res != CUDA_SUCCESS) {
        printf("cuModuleGetFunction() failed\n");
        return 0;
    }

    res = cuFuncSetBlockShape(func, sr->block_x, sr->block_y, 1);
    if (res != CUDA_SUCCESS) {
        printf("cuFuncSetBlockShape() failed\n");
        return 0;
    }

    cuParamSeti(func, 0, (unsigned long long)sr->din);
    cuParamSeti(func, 4, (unsigned long long)sr->din >> 32);
    cuParamSeti(func, 8, (unsigned long long)sr->dout);
    cuParamSeti(func, 12, (unsigned long long)sr->dout >> 32);
    cuParamSetSize(func, 16);

    res = cuLaunchGrid(func, sr->grid_x, sr->grid_y);
    if (res != CUDA_SUCCESS) {
        printf("cuLaunchGrid failed: res = %u\n", res);
        return 0;
    }

    return 0;
}

int test_prepare(struct kgpu_service_request *sr)
{
    cuMemcpyHtoD( (CUdeviceptr)sr->din, sr->hin, sr->insize );
    return 0;
}

int test_post(struct kgpu_service_request *sr)
{
    cuMemcpyDtoH( sr->hout, (CUdeviceptr)sr->dout, sr->outsize );
    return 0;
}

struct kgpu_service test_srv;

int init_service(void *lh, int (*reg_srv)(struct kgpu_service*, void*))
{
    CUresult res;
    printf("[libsrv_test] Info: init test service\n");
    
    sprintf(test_srv.name, "test_service");
    test_srv.sid = 0;
    test_srv.compute_size = test_compute_size;
    test_srv.launch = test_launch;
    test_srv.prepare = test_prepare;
    test_srv.post = test_post;


    res = cuModuleLoad(&module, "./test.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuModuleLoad() failed\n");
        return 0;
    }
    
    return reg_srv(&test_srv, lh);
}

int finit_service(void *lh, int (*unreg_srv)(const char*))
{
    printf("[libsrv_test] Info: finit test service\n");
    cuModuleUnload(module);
    return unreg_srv(test_srv.name);
}
