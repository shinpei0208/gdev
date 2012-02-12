/*
 * Copyright (C) 2011 Shinpei Kato
 *
 * Systems Research Lab, University of California at Santa Cruz
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

#include "cuda.h"
#include "gdev_cuda.h"

int gdev_device_count = 0;

/**
 * Returns in *major and *minor the major and minor revision numbers that 
 * define the compute capability of the device dev.
 *
 * Parameters:
 * major - Major revision number
 * minor - Minor revision number
 * dev - Device handle
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
	GDEV_PRINT("cuDeviceComputeCapability: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Returns in *device a device handle given an ordinal in the range 
 * [0, cuDeviceGetCount()-1].
 *
 * Parameters:
 * device - Returned device handle
 * ordinal	- Device number to get handle for
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (ordinal < 0 || ordinal >= gdev_device_count)
		return CUDA_ERROR_INVALID_VALUE;

	*device = ordinal;

	return CUDA_SUCCESS;

}

/**
 * Returns in *pi the integer value of the attribute attrib on device dev. 
 * The supported attributes are:
 * CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: 
 * Maximum number of threads per block;
 * CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
 * Maximum x-dimension of a block;
 * CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: 
 * Maximum y-dimension of a block;
 * CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: 
 * Maximum z-dimension of a block;
 * CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: 
 * Maximum x-dimension of a grid;
 * CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: 
 * Maximum y-dimension of a grid;
 * CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
 * Maximum z-dimension of a grid;
 * CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: 
 * Maximum amount of shared memory available to a thread block in bytes; 
 * this amount is shared by all thread blocks simultaneously resident on a 
 * multiprocessor;
 * CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
 * Memory available on device for __constant__ variables in a CUDA C kernel 
 * in bytes;
 * CU_DEVICE_ATTRIBUTE_WARP_SIZE: 
 * Warp size in threads;
 * CU_DEVICE_ATTRIBUTE_MAX_PITCH: 
 * Maximum pitch in bytes allowed by the memory copy functions that involve 
 * memory regions allocated through cuMemAllocPitch();
 * CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: 
 * Maximum number of 32-bit registers available to a thread block; 
 * this number is shared by all thread blocks simultaneously resident on a 
 * multiprocessor;
 * CU_DEVICE_ATTRIBUTE_CLOCK_RATE: 
 * Peak clock frequency in kilohertz;
 * CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: 
 * Alignment requirement; texture base addresses aligned to textureAlign 
 * bytes do not need an offset applied to texture fetches;
 * CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: 
 * 1 if the device can concurrently copy memory between host and device while
 * executing a kernel, or 0 if not;
 * CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: 
 * Number of multiprocessors on the device;
 * CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 
 * 1 if there is a run time limit for kernels executed on the device, or 0 if
 * not;
 * CU_DEVICE_ATTRIBUTE_INTEGRATED: 
 * 1 if the device is integrated with the memory subsystem, or 0 if not;
 * CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 
 * 1 if the device can map host memory into the CUDA address space, or 0 if not;
 * CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
 * Compute mode that device is currently in. Available modes are as follows:
 * CU_COMPUTEMODE_DEFAULT: 
 * Default mode - Device is not restricted and can have multiple CUDA contexts
 * present at a single time.
 * CU_COMPUTEMODE_EXCLUSIVE:
 * Compute-exclusive mode - Device can have only one CUDA context present on 
 * it at a time.
 * CU_COMPUTEMODE_PROHIBITED:
 * Compute-prohibited mode - Device is prohibited from creating new CUDA 
 * contexts.
 *
 * Parameters:
 * pi - Returned device attribute value
 * attrib - Device attribute to query
 * dev - Device handle
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
	GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Returns in *count the number of devices with compute capability greater
 * than or equal to 1.0 that are available for execution. If there is no such
 * device, cuDeviceGetCount() returns 0.
 *
 * Parameters:
 * count - Returned number of compute-capable devices
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuDeviceGetCount(int *count)
{
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!count)
		return CUDA_ERROR_INVALID_VALUE;

	*count = gdev_device_count;

	return CUDA_SUCCESS;
}

/**
 * Returns an ASCII string identifying the device dev in the NULL-terminated
 * string pointed to by name. len specifies the maximum length of the string 
 * that may be returned.
 *
 * Parameters:
 * name - Returned identifier string for the device
 * len - Maximum length of string to store in name
 * dev - Device to get identifier string for
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
	GDEV_PRINT("cuDeviceGetName: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Returns in *prop the properties of device dev. The CUdevprop structure is
 * defined as:
 *
 * typedef struct CUdevprop_st { 
 *   int maxThreadsPerBlock; 
 *   int maxThreadsDim[3];
 *   int maxGridSize[3]; 
 *   int sharedMemPerBlock;
 *   int totalConstantMemory;
 *   int SIMDWidth;
 *   int memPitch;
 *   int regsPerBlock;
 *   int clockRate;
 *   int textureAlign
 * } CUdevprop;
 *
 * where:
 * maxThreadsPerBlock is the maximum number of threads per block;
 * maxThreadsDim[3] is the maximum sizes of each dimension of a block;
 * maxGridSize[3] is the maximum sizes of each dimension of a grid;
 * sharedMemPerBlock is the total amount of shared memory available per block
 * in bytes;
 * totalConstantMemory is the total amount of constant memory available on the
 * device in bytes;
 * SIMDWidth is the warp size;
 * memPitch is the maximum pitch allowed by the memory copy functions that 
 * involve memory regions allocated through cuMemAllocPitch();
 * regsPerBlock is the total number of registers available per block;
 * clockRate is the clock frequency in kilohertz;
 * textureAlign is the alignment requirement; texture base addresses that are
 *  aligned to textureAlign bytes do not need an offset applied to texture 
 * fetches.
 *
 * Parameters:
 * prop - Returned properties of device
 * dev - Device to get properties for
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
	GDEV_PRINT("cuDeviceGetProperties: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Returns in *bytes the total amount of memory available on the device dev 
 * in bytes.
 *
 * Parameters:
 * bytes - Returned memory available on device in bytes
 * dev - Device handle
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
	Ghandle handle;
	uint64_t total_mem;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!bytes)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (gquery(handle, GDEV_QUERY_DEVICE_MEM_SIZE, &total_mem)) {
		return CUDA_ERROR_UNKNOWN;
	}

	*bytes = total_mem;

	return CUDA_SUCCESS;
}

