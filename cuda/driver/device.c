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
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/stat.h>
#include <linux/fcntl.h>
#include <linux/namei.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

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
	CUcontext ctx;
	gdev_list_for_each(ctx, &gdev_ctx_list, list_entry) {
		if (ctx->minor == minor) {
			handle = ctx->gdev_handle;
			break;
		}
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
	CUcontext ctx;
	gdev_list_for_each(ctx, &gdev_ctx_list, list_entry) {
		if (ctx->gdev_handle == handle) {
			return 0;
		}
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
	Ghandle handle;
	uint64_t chipset;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!major)
		return CUDA_ERROR_INVALID_VALUE;
	if (!minor)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen((int)dev);
	if (handle == NULL)
		return CUDA_ERROR_INVALID_DEVICE;

	if (gquery(handle, GDEV_QUERY_CHIPSET, &chipset)) {
		__gclose(handle);
		return CUDA_ERROR_UNKNOWN;
	}
	__gclose(handle);

	switch (chipset) {
	case 0x0c0:
	case 0x0c8:
		*major = 2;
		*minor = 0;
		break;
	case 0x0c1:
	case 0x0c3:
	case 0x0c4:
	case 0x0ce:
	case 0x0cf:
	case 0x0d9:
		*major = 2;
		*minor = 1;
		break;
	case 0x0e4:
	case 0x0e6:
	case 0x0e7:
		*major = 3;
		*minor = 0;
		break;
	case 0x0f0:
	case 0x108:
		*major = 3;
		*minor = 5;
		break;
	default:
		return CUDA_ERROR_INVALID_DEVICE;
	}

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
	Ghandle handle;
	uint64_t chipset;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	handle = __gopen((int)dev);
	if (handle == NULL)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gquery(handle, GDEV_QUERY_CHIPSET, &chipset)) {
		__gclose(handle);
		return CUDA_ERROR_UNKNOWN;
	}

	switch (attrib) {
	case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
	case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
	case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
	case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
		{
			uint64_t pci_vendor, pci_device, mp_count;
			struct device_properties *props;
			*pi = 0;
			if (gquery(handle, GDEV_QUERY_PCI_VENDOR, &pci_vendor))
				res = CUDA_ERROR_INVALID_DEVICE;
			else {
				if (pci_vendor != 0x10de)
					res = CUDA_ERROR_INVALID_DEVICE;
			}
			if (res != CUDA_SUCCESS)
				break;
			if (gquery(handle, GDEV_QUERY_PCI_DEVICE, &pci_device))
				res = CUDA_ERROR_INVALID_DEVICE;
			else {
				props = get_device_properties(pci_device);
				if (!props)
					res = CUDA_ERROR_INVALID_DEVICE;
				else
					switch (attrib) {
					case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
						*pi = props->mp_count;
						break;
					case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
						*pi = props->ecc;
						break;
					case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
						if (pci_device == 0x0E22 ||
						    pci_device == 0x0E24 ||
						    pci_device == 0x1205) {
							size_t mem_size = 0;
							cuDeviceTotalMem(&mem_size, dev);
							if (mem_size <= 768 * 1024 * 1024)
								*pi = 192;
							else
								*pi = 256;
							break;
						}
						*pi = props->bus_width;
						break;
					case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
						*pi = props->l2_cache;
						break;
					default:
						break;
					}
			}
			if (res != CUDA_SUCCESS)
				break;
			if (*pi)
				break;
			if (attrib == CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
				break;
			if (attrib == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) {
				if (gquery(handle,
					   GDEV_NVIDIA_QUERY_MP_COUNT,
					   &mp_count)) {
					res = CUDA_ERROR_INVALID_DEVICE;
					break;
				} else
					*pi = mp_count;
			}
			if (!*pi)
				GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1024;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
    	case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1024;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 64;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
			*pi = 65535;
			break;
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 2147483647;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
    	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 65535;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 49152;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 65536;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 32;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 2147483647;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
			*pi = 32768;
			break;
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 65536;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 512;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_INTEGRATED:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 0;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = CU_COMPUTEMODE_DEFAULT;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 65536;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
			*pi = 65535;
			break;
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 65536;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
			*pi = 2048;
			break;
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 4096;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH:
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 16384;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 2048;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 0;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
			*pi = 1536;
			break;
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 2048;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
		switch (chipset & 0x1f0) {
		case 0x0c0:
		case 0x0d0:
		case 0x0e0:
		case 0x0f0:
		case 0x100:
			*pi = 1;
			break;
		default:
			GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet\n");
			break;
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
    	case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
		{
			int phys = (int)dev;
			int bus_id[16];
			int dev_id[16];
			int i, j, k;
			int nvrm = 0;
			int no = 0;
#ifdef __KERNEL__
			struct file *file;
			struct path path;
#else
			int fd;
			struct stat sbuf;
#endif
			union {
				char cbuf[256];
				uint16_t ubuf[128];
			} buf;
			char name[256];
#ifdef __KERNEL__
			if (kern_path("/proc/gdev", LOOKUP_DIRECTORY, &path) == 0) {
#else
			if (stat("/proc/gdev",&sbuf) == 0) {
#endif
				snprintf(name, sizeof(name),
					 "/proc/gdev/vd%d/phys", (int)dev);
#ifdef __KERNEL__
				file = filp_open(name, O_RDONLY, 0);
				if (!file) {
					res = CUDA_ERROR_INVALID_DEVICE;
					break;
				}
				file->f_op->read(file, buf.cbuf, sizeof(buf), &file->f_pos);
				filp_close(file, NULL);
#else
				fd = open(name, O_RDONLY);
				if (fd < 0) {
					res = CUDA_ERROR_INVALID_DEVICE;
					break;
				}
				read(fd, buf.cbuf, sizeof(buf));
				close(fd);
#endif
				sscanf(buf.cbuf, "%d", &phys);
#ifdef __KERNEL__
			} else if (kern_path("/proc/driver/nvidia", LOOKUP_DIRECTORY, &path) == 0) {
#else
			} else if (stat("/proc/driver/nvidia", &sbuf) == 0) {
#endif
				nvrm = 1;
			}
			for (i = 0; i < 0x100 && no <= phys; i++) {
				for (j = 0; j < 0x20 && no <= phys; j++) {
					for (k = 0; k < 0x8; k++) {
						snprintf(name, sizeof(name),
							 "/proc/bus/pci/%02x/%02x.%x",
							 i, j, k);
#ifdef __KERNEL__
						file = filp_open(name, O_RDONLY, 0);
						if (!file)
							break;
						file->f_op->read(file, buf.cbuf, sizeof(buf), &file->f_pos);
						filp_close(file, NULL);
#else
						fd = open(name, O_RDONLY);
						if (fd < 0)
							break;
						read(fd, buf.cbuf, sizeof(buf));
						close(fd);
#endif
						if (nvrm &&
						    buf.ubuf[0] != 0x10de)
							break;
						if (buf.ubuf[5] == 0x0300) {
							bus_id[no] = i;
							dev_id[no] = j;
							no++;
							break;
						}
					}
				}
			}
			switch (attrib) {
			case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
				*pi = bus_id[phys];
				break;
			case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
				*pi = dev_id[phys];
				break;
			default:
				break;
			}
		}
		break;
    	case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
		*pi = 0; /* FIXME */
		break;
	default:
		GDEV_PRINT("cuDeviceGetAttribute: Not Implemented Yet(%d)\n", attrib);
		break;
	}

	__gclose(handle);

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
	Ghandle handle;
	uint64_t pci_vendor, pci_device;
	struct device_properties *props;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!name)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen((int)dev);
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
	Ghandle handle;
	uint64_t pci_vendor, pci_device;
	struct device_properties *props;
	int val;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!prop)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen((int)dev);
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

	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
	prop->maxThreadsPerBlock = val; 
	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
	prop->maxThreadsDim[0] = val;
	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
	prop->maxThreadsDim[1] = val;
	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
	prop->maxThreadsDim[2] = val;
	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
	prop->maxGridSize[0] = val; 
	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
	prop->maxGridSize[1] = val; 
	val = 1;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
	prop->maxGridSize[2] = val; 
	val = 0;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
	prop->sharedMemPerBlock = val;
	val = 0;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
	prop->totalConstantMemory = val;
	val = 0;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
	prop->SIMDWidth = val;
	val = 0;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_PITCH, dev);
	prop->memPitch = val;
	val = 0;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
	prop->regsPerBlock = val;
	/* FIXME */
	prop->clockRate = 0;
	val = 0;
	cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
	prop->textureAlign = val;

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
	Ghandle handle;
	uint64_t total_mem;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_device_count)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!bytes)
		return CUDA_ERROR_INVALID_VALUE;

	handle = __gopen((int)dev);
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

