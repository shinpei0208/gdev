/*
 * Copyright (C)  Yusuke FUJII <emb.ubi.yukke@gmail.com>
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
#ifndef __GDEV_USCHED_MONITOR__
#define __GDEV_USCHED_MONITOR__

#include "gdev_lib.h"
#include "gdev_sched.h"
#include "gdev_nvidia.h"
#include <pthread.h>

#define GDEV_DEVICE_MAX_COUNT 32
#define VDEVICE_MAX_COUNT 16
#define GDEV_PERIOD_DEFAULT 30000 /* microseconds */

extern struct gdev_device *phys;
extern struct gdev_device *gdevs;
extern struct gdev_sched_entity *se;
extern struct gdev_mem *mem;
extern struct gdev_vas *vas;
extern int shmid;
extern int semid;
extern int msgid;
extern int gdev_count;
extern int *_gdev_vcount;
extern int semno;
extern int semid_array[];
extern int gdev_bw_set[];

void gdev_mutex_init(struct gdev_mutex *);
int init_gdev_monitor();

#endif 
