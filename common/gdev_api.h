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

#ifndef __GDEV_API_H__
#define __GDEV_API_H__

#ifdef __KERNEL__
#include "gdev_drv.h"
#else
#include "gdev_lib.h"
#endif
#include "gdev_nvidia_def.h"
/* add also:
 * #include "gdev_amd_def.h"
 * #include "gdev_intel_def.h"
 */

/**
 * Gdev APIs:
 */
extern gdev_handle_t *gopen(int);
extern int gclose(gdev_handle_t*);
extern uint64_t gmalloc(gdev_handle_t*, uint64_t);
extern int gfree(gdev_handle_t*, uint64_t);
extern int gmemcpy_from_device(gdev_handle_t*, void*, uint64_t, uint64_t);
extern int gmemcpy_user_from_device(gdev_handle_t*, void*, uint64_t, uint64_t);
extern int gmemcpy_to_device(gdev_handle_t*, uint64_t, void*, uint64_t);
extern int gmemcpy_user_to_device(gdev_handle_t*, uint64_t, void*, uint64_t);
extern int gmemcpy_in_device(gdev_handle_t*, uint64_t, uint64_t, uint64_t);
extern int glaunch(gdev_handle_t*, struct gdev_kernel*, uint32_t*);
extern void gsync(gdev_handle_t*, uint32_t);
extern int gquery(gdev_handle_t*, uint32_t, uint32_t*);
extern int gtune(gdev_handle_t*, uint32_t, uint32_t);

/**
 * tuning types for Gdev resource management parameters.
 */
#define GDEV_TUNE_MEMCPY_PIPELINE_COUNT 1
#define GDEV_TUNE_MEMCPY_CHUNK_SIZE 2

#endif
