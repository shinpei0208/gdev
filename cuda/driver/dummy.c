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

#include "cuda.h"
#include "gdev_cuda.h"

/*
 * Memory
 */
CUresult __attribute__((weak)) cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuArrayDestroy(CUarray hArray)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemGetInfo(size_t *free, size_t *total)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemHostGetFlags(unsigned int *pFlags, void *p)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemHostRegister(void *pp, unsigned long long bytesize, unsigned int Flags)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemHostUnregister(void *pp)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpy2D(const CUDA_MEMCPY2D *pCopy)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpy3D(const CUDA_MEMCPY3D *pCopy)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyAtoA(CUarray dstArray, unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyAtoH(void *dstHost, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyDtoA(CUarray dstArray, unsigned int dstIndex, CUdeviceptr srcDevice, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyHtoA(CUarray dstArray, unsigned int dstIndex, const void *pSrc, unsigned int ByteCount)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount, CUstream hStream)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemcpyHtoAAsync(CUarray dstArray, unsigned int dstIndex, const void *pSrc, unsigned int ByteCount, CUstream hStream)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, unsigned int N)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, unsigned int N)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, unsigned int N)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemsetD2D8(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemsetD2D16(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuMemsetD2D32(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height)
{
	return CUDA_SUCCESS;
}

/*
 * Execution
 */
CUresult __attribute__((weak)) cuFuncSetCacheConfig(CUfunction hFunc, CUfunc_cache config)
{
	return CUDA_SUCCESS;
}

/*
 * Graphics
 */
CUresult __attribute__((weak)) cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
{
	return CUDA_SUCCESS;
}

/*
 * Texture Reference
 */
CUresult __attribute__((weak)) cuTexRefCreate(CUtexref *pTexRef)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefDestroy(CUtexref hTexRef)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef)
{
	return CUDA_SUCCESS;
}

CUresult __attribute__((weak)) cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef)
{
	return CUDA_SUCCESS;
}

#if 0
/*
 *
 */
CUresult __attribute__((weak)) cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
	return CUDA_SUCCESS;
}
#endif

