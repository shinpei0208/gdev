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

#ifndef __GDEV_CONF_H__
#define __GDEV_CONF_H__

#include "gdev_nvidia.h"
// #include "gdev_amd.h"
// #include "gdev_intel.h"

#define GDEV_DEVICE_MAX_COUNT 32

#define GDEV_PIPELINE_MAX_COUNT 8
#define GDEV_PIPELINE_MIN_COUNT 1
#define GDEV_PIPELINE_DEFAULT_COUNT 2

#define GDEV_CHUNK_MAX_SIZE 0x2000000 /* 32MB */
#define GDEV_CHUNK_DEFAULT_SIZE 0x100000 /* 1MB */

/* define this if you want to allocate a new bounce buffer every time
   you copy data to/from device memory. */
//#define GDEV_NO_STATIC_BOUNCE_BUFFER

#define GDEV_SWAP_DEVICE_SIZE 0x2000000
#define GDEV_SWAP_DMA_SIZE 0x2000000

#define GDEV_MEMCPY_IORW_LIMIT 0x400 /* bytes */

#define GDEV_DEBUG_PRINT 0

#endif
