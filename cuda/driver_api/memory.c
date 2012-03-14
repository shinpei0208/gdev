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
#include "gdev_api.h"
#include "gdev_cuda.h"

/***************************************************************************
 * Currently a very limited set of memory management functions is supported.
 * There are lots things to be additionally implemented...
 ***************************************************************************/

/**
 * Allocates bytesize bytes of linear memory on the device and returns in 
 * @dptr a pointer to the allocated memory. The allocated memory is suitably 
 * aligned for any kind of variable. The memory is not cleared. If bytesize 
 * is 0, cuMemAlloc() returns CUDA_ERROR_INVALID_VALUE.
 *
 * Parameters:
 * dptr - Returned device pointer
 * bytesize - Requested allocation size in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_OUT_OF_MEMORY 
 */
CUresult cuMemAlloc(CUdeviceptr *dptr, unsigned int bytesize)
{
	Ghandle handle;
	uint64_t addr;
	uint64_t size = bytesize;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!dptr)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;
	if (!(addr = gmalloc(handle, size))) {
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	*dptr = addr;

	return CUDA_SUCCESS;
}

/**
 * Frees the memory space pointed to by dptr, which must have been returned 
 * by a previous call to cuMemAlloc() or cuMemAllocPitch().
 *
 * Parameters:
 * dptr 	- Pointer to memory to free
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemFree(CUdeviceptr dptr)
{
	Ghandle handle;
	uint64_t addr = dptr;
	uint64_t size;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	/* wait for all kernels to complete - some may be using the memory. */
	cuCtxSynchronize();

	handle = gdev_ctx_current->gdev_handle;

	if (!(size = gfree(handle, addr)))
		return CUDA_ERROR_INVALID_VALUE;

	return CUDA_SUCCESS;
}

/**
 * Allocates bytesize bytes of host memory that is page-locked and accessible 
 * to the device. The driver tracks the virtual memory ranges allocated with 
 * this function and automatically accelerates calls to functions such as 
 * cuMemcpy(). Since the memory can be accessed directly by the device, it can
 * be read or written with much higher bandwidth than pageable memory obtained
 * with functions such as malloc(). Allocating excessive amounts of memory 
 * with cuMemAllocHost() may degrade system performance, since it reduces the 
 * amount of memory available to the system for paging. As a result, this 
 * function is best used sparingly to allocate staging areas for data exchange
 * between host and device.
 *
 * Note all host memory allocated using cuMemHostAlloc() will automatically 
 * be immediately accessible to all contexts on all devices which support 
 * unified addressing (as may be queried using 
 * CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING). The device pointer that may be 
 * used to access this host memory from those contexts is always equal to the 
 * returned host pointer *pp. See Unified Addressing for additional details.
 *
 * Parameters:
 * pp - Returned host pointer to page-locked memory
 * bytesize - Requested allocation size in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_OUT_OF_MEMORY 
 */
CUresult cuMemAllocHost(void **pp, unsigned int bytesize)
{
	Ghandle handle;
	void *buf;
	uint64_t size = bytesize;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!pp)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;
	if (!(buf = gmalloc_dma(handle, size)))
		return CUDA_ERROR_OUT_OF_MEMORY;

	*pp = buf;

	return CUDA_SUCCESS;
}

/**
 * Frees the memory space pointed to by p, which must have been returned by a 
 * previous call to cuMemAllocHost().
 *
 * Parameters:
 * p - Pointer to memory to free
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemFreeHost(void *p)
{
	Ghandle handle;
	void *buf = p;
	uint64_t size;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;

	/* wait for all kernels to complete - some may be using the memory. */
	cuCtxSynchronize();

	handle = gdev_ctx_current->gdev_handle;

	if (!(size = gfree_dma(handle, buf)))
		return CUDA_ERROR_INVALID_VALUE;

	return CUDA_SUCCESS;
}

/**
 * Copies from host memory to device memory. dstDevice and srcHost are the base
 * addresses of the destination and source, respectively. ByteCount specifies
 * the number of bytes to copy. Note that this function is synchronous.
 *
 * Parameters:
 * dstDevice - Destination device pointer
 * srcHost - Source host pointer
 * ByteCount - Size of memory copy in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemcpyHtoD
(CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount)
{
	Ghandle handle;
	const void *src_buf = srcHost;
	uint64_t dst_addr = dstDevice;
	uint32_t size = ByteCount;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!src_buf || !dst_addr || !size)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (gmemcpy_to_device(handle, dst_addr, src_buf, size))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

/**
 * Copies from host memory to device memory. dstDevice and srcHost are the base
 * addresses of the destination and source, respectively. ByteCount specifies 
 * the number of bytes to copy.
 *
 * cuMemcpyHtoDAsync() is asynchronous and can optionally be associated to a 
 * stream by passing a non-zero hStream argument. It only works on page-locked 
 * memory and returns an error if a pointer to pageable memory is passed as 
 * input.
 *
 * Parameters:
 * dstDevice - Destination device pointer
 * srcHost - Source host pointer
 * ByteCount - Size of memory copy in bytes
 * hStream - Stream identifier 
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemcpyHtoDAsync
(CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, 
 CUstream hStream)
{
	GDEV_PRINT("cuMemcpyHtoD: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Copies from device to host memory. dstHost and srcDevice specify the base 
 * pointers of the destination and source, respectively. ByteCount specifies 
 * the number of bytes to copy. Note that this function is synchronous.
 *
 * Parameters:
 * dstHost - Destination host pointer
 * srcDevice - Source device pointer
 * ByteCount - Size of memory copy in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemcpyDtoH
(void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount)
{
	Ghandle handle;
	void *dst_buf = dstHost;
	uint64_t src_addr = srcDevice;
	uint32_t size = ByteCount;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!dst_buf || !src_addr || !size)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (gmemcpy_from_device(handle, dst_buf, src_addr, size))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

/**
 * Copies from device to host memory. dstHost and srcDevice specify the base 
 * pointers of the destination and source, respectively. ByteCount specifies 
 * the number of bytes to copy.
 *
 * cuMemcpyDtoHAsync() is asynchronous and can optionally be associated to a 
 * stream by passing a non-zero hStream argument. It only works on page-locked 
 * memory and returns an error if a pointer to pageable memory is passed as 
 * input.
 *
 * Parameters:
 * dstHost - Destination host pointer
 * srcDevice - Source device pointer
 * ByteCount - Size of memory copy in bytes
 * hStream - Stream identifier
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemcpyDtoHAsync
(void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hstream)
{
	GDEV_PRINT("cuMemcpyDtoH: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Copies from device memory to device memory. dstDevice and srcDevice are the 
 * base pointers of the destination and source, respectively. ByteCount 
 * specifies the number of bytes to copy. Note that this function is 
 * asynchronous.
 *
 * Parameters:
 * dstDevice - Destination device pointer
 * srcDevice - Source device pointer
 * ByteCount - Size of memory copy in bytes
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount)
{
	Ghandle handle;
	uint64_t dst_addr = dstDevice;
	uint64_t src_addr = srcDevice;
	uint32_t size = ByteCount;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!dst_addr || !src_addr || !size)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (gmemcpy_in_device(handle, dst_addr, src_addr, size))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

/**
 * Allocates bytesize bytes of host memory that is page-locked and accessible 
 * to the device. The driver tracks the virtual memory ranges allocated with 
 * this function and automatically accelerates calls to functions such as 
 * cuMemcpyHtoD(). Since the memory can be accessed directly by the device, it 
 * can be read or written with much higher bandwidth than pageable memory 
 * obtained with functions such as malloc(). Allocating excessive amounts of 
 * pinned memory may degrade system performance, since it reduces the amount of 
 * memory available to the system for paging. As a result, this function is 
 * best used sparingly to allocate staging areas for data exchange between host
 * and device.
 *
 * The Flags parameter enables different options to be specified that affect 
 * the allocation, as follows.
 *
 * CU_MEMHOSTALLOC_PORTABLE: The memory returned by this call will be 
 * considered as pinned memory by all CUDA contexts, not just the one that 
 * performed the allocation.
 *
 * CU_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the CUDA address space. 
 * The device pointer to the memory may be obtained by calling 
 * cuMemHostGetDevicePointer(). This feature is available only on GPUs with 
 * compute capability greater than or equal to 1.1.
 *
 * CU_MEMHOSTALLOC_WRITECOMBINED: Allocates the memory as write-combined (WC). 
 * WC memory can be transferred across the PCI Express bus more quickly on some
 * system configurations, but cannot be read efficiently by most CPUs. WC 
 * memory is a good option for buffers that will be written by the CPU and read
 * by the GPU via mapped pinned memory or host->device transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate 
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * The CUDA context must have been created with the CU_CTX_MAP_HOST flag in 
 * order for the CU_MEMHOSTALLOC_MAPPED flag to have any effect.
 *
 * The CU_MEMHOSTALLOC_MAPPED flag may be specified on CUDA contexts for 
 * devices that do not support mapped pinned memory. The failure is deferred to
 * cuMemHostGetDevicePointer() because the memory may be mapped into other CUDA
 * contexts via the CU_MEMHOSTALLOC_PORTABLE flag.
 *
 * The memory allocated by this function must be freed with cuMemFreeHost().
 *
 * Note all host memory allocated using cuMemHostAlloc() will automatically be
 * immediately accessible to all contexts on all devices which support unified 
 * addressing (as may be queried using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
 * Unless the flag CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device 
 * pointer that may be used to access this host memory from those contexts is 
 * always equal to the returned host pointer *pp. If the flag 
 * CU_MEMHOSTALLOC_WRITECOMBINED is specified, then the function 
 * cuMemHostGetDevicePointer() must be used to query the device pointer, even 
 * if the context supports unified addressing. See Unified Addressing for 
 * additional details.
 *
 * Parameters:
 * pp - Returned host pointer to page-locked memory
 * bytesize - Requested allocation size in bytes
 * Flags - Flags for allocation request
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_OUT_OF_MEMORY 
 */
CUresult cuMemHostAlloc(void **pp, unsigned int bytesize, unsigned int Flags)
{
	if (Flags & CU_MEMHOSTALLOC_PORTABLE) {
		GDEV_PRINT("CU_MEMHOSTALLOC_PORTABLE: Not Implemented Yet\n");
		return CUDA_ERROR_UNKNOWN;
	}

	if (Flags & CU_MEMHOSTALLOC_WRITECOMBINED) {
		GDEV_PRINT("CU_MEMHOSTALLOC_WRITECOMBINED: Not Implemented Yet\n");
		return CUDA_ERROR_UNKNOWN;
	}

	/* our implementation uses CU_MEMHOSTALLOC_DEVICEMAP by default. */
	return cuMemAllocHost(pp, bytesize);
}

/**
 * Passes back the device pointer pdptr corresponding to the mapped, pinned 
 * host buffer p allocated by cuMemHostAlloc.
 *
 * cuMemHostGetDevicePointer() will fail if the CU_MEMALLOCHOST_DEVICEMAP flag 
 * was not specified at the time the memory was allocated, or if the function 
 * is called on a GPU that does not support mapped pinned memory.
 *
 * Flags provides for future releases. For now, it must be set to 0.
 *
 * Parameters:
 * pdptr - Returned device pointer
 * p - Host pointer
 * Flags - Options (must be 0)
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
	Ghandle handle;
	uint64_t addr;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!pdptr || !p || Flags != 0)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;
	addr = gvirtget(handle, p);
	*pdptr = (CUdeviceptr)addr;

	return CUDA_SUCCESS;
}

/**
 * Gdev extension: maps device memory to host memory.
 *
 * Parameters:
 * dptr - Device pointer
 * buf - Pointer to user buffer
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemMap(void **buf, CUdeviceptr dptr, unsigned int bytesize)
{
	Ghandle handle;
	uint64_t addr = dptr;
	void *map;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!addr || !buf || !bytesize)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (!(map = gmap(handle, addr, bytesize)))
		return CUDA_ERROR_UNKNOWN;

	*buf = map;

	return CUDA_SUCCESS;
}

/**
 * Gdev extension: unmaps device memory from host memory.
 *
 * Parameters:
 * buf - User buffer
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemUnmap(void *buf)
{
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!buf)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (gunmap(handle, buf))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}

/**
 * Gdev extension: returns physical bus address associated to user buffer.
 * Note that the address is contiguous only within the page boundary.
 *
 * Parameters:
 * addr - Physical bus address obtained
 * p - Pointer to user buffer
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */
CUresult cuMemGetPhysAddr(unsigned long long *addr, void *p)
{
	Ghandle handle;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!gdev_ctx_current)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!addr || !p)
		return CUDA_ERROR_INVALID_VALUE;

	handle = gdev_ctx_current->gdev_handle;

	if (!(*addr = gphysget(handle, p)))
		return CUDA_ERROR_UNKNOWN;

	return CUDA_SUCCESS;
}
