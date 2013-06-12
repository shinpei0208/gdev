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
#include "device.h"

int gdev_device_count = 0;

static Ghandle __gopen(int minor)
{
#if 0
	return gopen(minor);
#else
	Ghandle handle = NULL;
	CUresult res;
	CUcontext ctx;
	res = cuCtxGetCurrent(&ctx);
	if (res == CUDA_SUCCESS) {
		if (ctx)
			handle = ctx->gdev_handle;
	}
	if (handle == NULL)
		handle = gopen(minor);
	return handle;
#endif
}
static int __gclose(Ghandle handle)
{
#if 0
	return gclose(handle);
#else
	CUresult res;
	CUcontext ctx;
	res = cuCtxGetCurrent(&ctx);
	if (res == CUDA_SUCCESS) {
		if (ctx && handle == ctx->gdev_handle)
				return 0;
	}
	return gclose(handle);
#endif
}

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
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!major)
		return CUDA_ERROR_INVALID_VALUE;
	if (!minor)
		return CUDA_ERROR_INVALID_VALUE;

	*major = 2;
	*minor = 0;

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
	int res = CUDA_SUCCESS;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	switch (attrib) {
	case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
		{
			uint64_t pci_vendor, pci_device, mp_count;
			int minor = (int)dev;
			Ghandle handle;
			struct device_properties *props;

			handle = __gopen(minor);
			if (handle == NULL) {
				res = CUDA_ERROR_INVALID_CONTEXT;
				break;
			}
			*pi = 0;
#if !defined(__KERNEL__) || (LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0))
			if (gquery(handle, GDEV_QUERY_PCI_VENDOR, &pci_vendor))
				res = CUDA_ERROR_INVALID_DEVICE;
			else {
				if (pci_vendor != 0x10de)
					res = CUDA_ERROR_INVALID_DEVICE;
			}
			if (res != CUDA_SUCCESS) {
				__gclose(handle);
				break;
			}
			if (gquery(handle, GDEV_QUERY_PCI_DEVICE, &pci_device))
				res = CUDA_ERROR_INVALID_DEVICE;
			else {
				props = get_device_properties(pci_device);
				if (!props)
					res = CUDA_ERROR_INVALID_DEVICE;
				else
					*pi = props->mp_count;
			}
			if (res != CUDA_SUCCESS) {
				__gclose(handle);
				break;
			}
#endif
			if (!*pi) {
				if (gquery(handle, GDEV_NVIDIA_QUERY_MP_COUNT,
					&mp_count))
					res = CUDA_ERROR_INVALID_DEVICE;
				else
					*pi = mp_count;
			}
			__gclose(handle);
		}
		break;
	case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
		{
			uint64_t pci_device;
			int minor = (int)dev;
			Ghandle handle;

			handle = __gopen(minor);
			if (handle == NULL) {
				res = CUDA_ERROR_INVALID_CONTEXT;
				break;
			}
			if (gquery(handle, GDEV_QUERY_PCI_DEVICE, &pci_device))
				res = CUDA_ERROR_INVALID_DEVICE;
			else
				*pi = pci_device;
			__gclose(handle);
		}
		break;
	default:
		GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
		break;
	}

	return res;
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
	int minor = (int)dev;
	Ghandle handle;
	uint64_t pci_vendor, pci_device;
	struct device_properties *props;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!name)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen(minor);
	if (handle == NULL)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gquery(handle, GDEV_QUERY_PCI_VENDOR, &pci_vendor)) {
		__gclose(handle);
		return CUDA_ERROR_UNKNOWN;
	}
	if (pci_vendor != 0x10de) {
		name[0] = '\0';
		goto end;
	}

	if (gquery(handle, GDEV_QUERY_PCI_DEVICE, &pci_device)) {
		__gclose(handle);
		return CUDA_ERROR_UNKNOWN;
	}
	props = get_device_properties(pci_device);
	if (!props) {
		name[0] = '\0';
		goto end;
	}
	strncpy(name, props->name, len-1);
	name[len-1] = '\0';

end:
	__gclose(handle);

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
	int minor = (int)dev;
	Ghandle handle;
	uint64_t pci_vendor, pci_device;
	struct device_properties *props;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!prop)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen(minor);
	if (handle == NULL)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gquery(handle, GDEV_QUERY_PCI_VENDOR, &pci_vendor)) {
		return CUDA_ERROR_UNKNOWN;
	}
	if (pci_vendor != 0x10de) {
		goto unknown;
	}

	if (gquery(handle, GDEV_QUERY_PCI_DEVICE, &pci_device)) {
		return CUDA_ERROR_UNKNOWN;
	}
	props = get_device_properties(pci_device);
	if (!props) {
		goto unknown;
	}

	prop->maxThreadsPerBlock = 1; 
	prop->maxThreadsDim[0] = 1;
	prop->maxThreadsDim[1] = 1;
	prop->maxThreadsDim[2] = 1;
	prop->maxGridSize[0] = 1; 
	prop->maxGridSize[1] = 1; 
	prop->maxGridSize[2] = 1; 
	prop->sharedMemPerBlock = 0;
	prop->totalConstantMemory = 0;
	prop->SIMDWidth = 0;
	prop->memPitch = 0;
	prop->regsPerBlock = 0;
	prop->clockRate = 0;
	prop->textureAlign = 0;

	goto end;

unknown:
	prop->maxThreadsPerBlock = 1; 
	prop->maxThreadsDim[0] = 1;
	prop->maxThreadsDim[1] = 1;
	prop->maxThreadsDim[2] = 1;
	prop->maxGridSize[0] = 1; 
	prop->maxGridSize[1] = 1; 
	prop->maxGridSize[2] = 1; 
	prop->sharedMemPerBlock = 0;
	prop->totalConstantMemory = 0;
	prop->SIMDWidth = 0;
	prop->memPitch = 0;
	prop->regsPerBlock = 0;
	prop->clockRate = 0;
	prop->textureAlign = 0;

end:
	__gclose(handle);

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
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
	int minor = (int)dev;
	Ghandle handle;
	uint64_t total_mem;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!bytes)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen(minor);
	if (handle == NULL)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gquery(handle, GDEV_QUERY_DEVICE_MEM_SIZE, &total_mem)) {
		__gclose(handle);
		return CUDA_ERROR_UNKNOWN;
	}

	*bytes = total_mem;

	__gclose(handle);

	return CUDA_SUCCESS;
}
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
	return cuDeviceTotalMem_v2(bytes, dev);
}

