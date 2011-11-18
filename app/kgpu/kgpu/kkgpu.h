/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Internal header used by KGPU only
 *
 */

#ifndef ___KKGPU_H__
#define ___KKGPU_H__

#include "kgpu.h"
#include <linux/types.h>

#define kgpu_log(level, ...) kgpu_do_log(level, "kgpu", ##__VA_ARGS__)
#define dbg(...) kgpu_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)

/*
 * Buffer management stuff, put them here in case we may
 * create a kgpu_buf.c for buffer related functions.
 */
#define KGPU_BUF_UNIT_SIZE (1024*1024)
#define KGPU_BUF_NR_FRAMES_PER_UNIT (KGPU_BUF_UNIT_SIZE/PAGE_SIZE)

/* memory ops */
extern struct page *kgpu_v2page(unsigned long vaddr);

#endif
