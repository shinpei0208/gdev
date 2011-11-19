/*
* Copyright 2011 Shinpei Kato
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

#ifndef __GDEV_LIB_H__
#define __GDEV_LIB_H__

#include <errno.h> /* ENOMEN, etc. */
#include <sched.h> /* sched_yield, etc. */
#include <stdint.h> /* uint32_t, etc.*/
#include <stdio.h> /* printf, etc. */
#include <stdlib.h> /* malloc/free, etc. */
#include <string.h> /* memcpy, etc. */

#define GDEV_DEV_GET(handle) (handle)->gdev
#define GDEV_DEV_SET(handle, dev) (handle)->gdev = (dev)
#define GDEV_VAS_GET(handle) (handle)->vas
#define GDEV_VAS_SET(handle, vas) (handle)->vas = (vas)
#define GDEV_CTX_GET(handle) (handle)->ctx
#define GDEV_CTX_SET(handle, ctx) (handle)->ctx = (ctx)
#define GDEV_DMA_GET(handle) (handle)->dma_mem
#define GDEV_DMA_SET(handle, mem) (handle)->dma_mem = (mem)
#define GDEV_PIPELINE_GET(handle) (handle)->pipeline_count
#define GDEV_PIPELINE_SET(handle, val) (handle)->pipeline_count = val
#define GDEV_CHUNK_GET(handle) (handle)->chunk_size
#define GDEV_CHUNK_SET(handle, val) (handle)->chunk_size = val
#define GDEV_MINOR_GET(handle) (handle)->dev_id 
#define GDEV_MINOR_SET(handle, val) (handle)->dev_id = val
#define GDEV_PRINT(fmt, arg...) fprintf(stderr, "[gdev] " fmt, ##arg)
#define GDEV_DPRINT(fmt, arg...)					\
	if (GDEV_DEBUG_PRINT)							\
		fprintf(stderr, "[gdev:debug] " fmt, ##arg)

#define MALLOC(x) malloc(x)
#define FREE(x) free(x)
#define SCHED_YIELD() sched_yield()
#define MB() //mb()
#define COPY_FROM_USER(dst, src, size) memcpy(dst, src, size)
#define COPY_TO_USER(dst, src, size) memcpy(dst, src, size)

typedef int gdev_handle_t;

#endif
