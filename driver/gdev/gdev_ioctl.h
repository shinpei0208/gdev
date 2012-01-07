/*
 * Copyright 2011 Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
 *
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

#ifndef __GDEV_IOCTL_H__
#define __GDEV_IOCTL_H__

#include "gdev_api.h"
#include "gdev_ioctl_def.h"

int gdev_ioctl_gmalloc(Ghandle h, unsigned long arg);
int gdev_ioctl_gfree(Ghandle h, unsigned long arg);
int gdev_ioctl_gmalloc_dma(Ghandle h, unsigned long arg);
int gdev_ioctl_gfree_dma(Ghandle h, unsigned long arg);
int gdev_ioctl_gmemcpy_to_device(Ghandle h, unsigned long arg);
int gdev_ioctl_gmemcpy_to_device_async(Ghandle h, unsigned long arg);
int gdev_ioctl_gmemcpy_from_device(Ghandle h, unsigned long arg);
int gdev_ioctl_gmemcpy_from_device_async(Ghandle h, unsigned long arg);
int gdev_ioctl_gmemcpy_in_device(Ghandle h, unsigned long arg);
int gdev_ioctl_glaunch(Ghandle h, unsigned long arg);
int gdev_ioctl_gsync(Ghandle h, unsigned long arg);
int gdev_ioctl_gquery(Ghandle h, unsigned long arg);
int gdev_ioctl_gtune(Ghandle h, unsigned long arg);
int gdev_ioctl_gshmget(Ghandle h, unsigned long arg);
int gdev_ioctl_gshmat(Ghandle h, unsigned long arg);
int gdev_ioctl_gshmdt(Ghandle h, unsigned long arg);
int gdev_ioctl_gshmctl(Ghandle h, unsigned long arg);

#endif
