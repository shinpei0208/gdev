/*
 * Copyright (C) Shinpei Kato
 *
 * University of California at Santa Cruz
 * Systems Research Lab.
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
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/stat.h>
#include "cuda.h"

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("Gdev/KCUDA");
MODULE_AUTHOR("Shinpei Kato");

#define MODULE_NAME	"kcuda"

static int __init kcuda_module_init(void)
{
	printk("[kcuda] Gdev/KCUDA loaded\n");
	return 0;
}

static void __exit kcuda_module_exit(void)
{
	printk("[kcuda] Gdev/KCUDA unloaded\n");
}

module_init(kcuda_module_init);
module_exit(kcuda_module_exit);

/* Initialization */
EXPORT_SYMBOL(cuInit);
/* Device Management */
EXPORT_SYMBOL(cuDeviceComputeCapability);
EXPORT_SYMBOL(cuDeviceGet);
EXPORT_SYMBOL(cuDeviceGetAttribute);
EXPORT_SYMBOL(cuDeviceGetCount);
EXPORT_SYMBOL(cuDeviceGetName);
EXPORT_SYMBOL(cuDeviceGetProperties);
EXPORT_SYMBOL(cuDeviceTotalMem);
/* Version Management */
EXPORT_SYMBOL(cuDriverGetVersion);
/* Context Management */
EXPORT_SYMBOL(cuCtxAttach);
EXPORT_SYMBOL(cuCtxCreate);
EXPORT_SYMBOL(cuCtxDestroy);
EXPORT_SYMBOL(cuCtxDetach);
EXPORT_SYMBOL(cuCtxGetDevice);
EXPORT_SYMBOL(cuCtxPopCurrent);
EXPORT_SYMBOL(cuCtxPushCurrent);
EXPORT_SYMBOL(cuCtxSynchronize);
/* Module Management */
EXPORT_SYMBOL(cuModuleGetFunction);
EXPORT_SYMBOL(cuModuleGetGlobal);
EXPORT_SYMBOL(cuModuleGetTexRef);
EXPORT_SYMBOL(cuModuleLoad);
EXPORT_SYMBOL(cuModuleLoadData);
EXPORT_SYMBOL(cuModuleLoadDataEx);
EXPORT_SYMBOL(cuModuleLoadFatBinary);
EXPORT_SYMBOL(cuModuleUnload);
/* Execution Control */
EXPORT_SYMBOL(cuFuncGetAttribute);
EXPORT_SYMBOL(cuFuncSetBlockShape);
EXPORT_SYMBOL(cuFuncSetSharedSize);
EXPORT_SYMBOL(cuLaunch);
EXPORT_SYMBOL(cuLaunchGrid);
EXPORT_SYMBOL(cuLaunchGridAsync);
EXPORT_SYMBOL(cuParamSetf);
EXPORT_SYMBOL(cuParamSeti);
EXPORT_SYMBOL(cuParamSetSize);
EXPORT_SYMBOL(cuParamSetTexRef);
EXPORT_SYMBOL(cuParamSetv);
/* Memory Management (Incomplete) */
EXPORT_SYMBOL(cuMemAlloc);
EXPORT_SYMBOL(cuMemFree);
EXPORT_SYMBOL(cuMemAllocHost);
EXPORT_SYMBOL(cuMemFreeHost);
EXPORT_SYMBOL(cuMemcpyDtoH);
EXPORT_SYMBOL(cuMemcpyDtoHAsync);
EXPORT_SYMBOL(cuMemcpyHtoD);
EXPORT_SYMBOL(cuMemcpyHtoDAsync);
EXPORT_SYMBOL(cuMemcpyDtoD);
EXPORT_SYMBOL(cuMemHostAlloc);
EXPORT_SYMBOL(cuMemHostGetDevicePointer);
/* Memory mapping - Gdev extension */
EXPORT_SYMBOL(cuMemMap);
EXPORT_SYMBOL(cuMemUnmap);
/* Memory mapped address - Gdev extension */
EXPORT_SYMBOL(cuMemGetPhysAddr);

/* Stream Management */
EXPORT_SYMBOL(cuStreamCreate);
EXPORT_SYMBOL(cuStreamDestroy);
EXPORT_SYMBOL(cuStreamQuery);
EXPORT_SYMBOL(cuStreamSynchronize);
EXPORT_SYMBOL(cuStreamWaitEvent);

/* Inter-Process Communication (IPC) - Gdev extension */
EXPORT_SYMBOL(cuShmGet);
EXPORT_SYMBOL(cuShmAt);
EXPORT_SYMBOL(cuShmDt);
EXPORT_SYMBOL(cuShmCtl);
