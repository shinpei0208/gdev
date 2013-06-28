/*
 * Copyright (C) Shinpei Kato
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

#ifndef __GDEV_CONF_H__
#define __GDEV_CONF_H__

#include "gdev_autogen.h"
#include "gdev_nvidia.h"
// #include "gdev_amd.h"
// #include "gdev_intel.h"

#define GDEV_CONTEXT_MAX_COUNT 128 /* # of GPU contexts */
#define GDEV_CONTEXT_LIMIT 16

#define GDEV_PIPELINE_MAX_COUNT 4
#define GDEV_PIPELINE_MIN_COUNT 1
#define GDEV_PIPELINE_DEFAULT_COUNT 2

#define GDEV_CHUNK_MAX_SIZE 0x2000000 /* 32MB */
#define GDEV_CHUNK_DEFAULT_SIZE 0x40000 /* 256KB */

#define GDEV_SWAP_MEM_SIZE 0x8000000 /* 128MB */

#define GDEV_MEMCPY_IOREAD_LIMIT 0x1000 /* 4KB */
#define GDEV_MEMCPY_IOWRITE_LIMIT 0x400000 /* 4MB */

#define GDEV0_VIRTUAL_DEVICE_COUNT 4 /* # of virtual devices */
#define GDEV1_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */
#define GDEV2_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */
#define GDEV3_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */
#define GDEV4_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */
#define GDEV5_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */
#define GDEV6_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */
#define GDEV7_VIRTUAL_DEVICE_COUNT 0 /* # of virtual devices */

// #define GDEV_DEBUG_PRINT
// #define GDEV_SCHED_DISABLED

#endif
