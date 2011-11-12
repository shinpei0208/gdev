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
#include <fcntl.h>
#include <stdio.h>
#include <sys/unistd.h>

int gdev_initialized = 0;
int gdev_device_count = 0;

/**
 * Initializes the driver API and must be called before any other function
 * from the driver API. Currently, the Flags parameter must be 0. If cuInit()
 * has not been called, any function from the driver API will return 
 * CUDA_ERROR_NOT_INITIALIZED.
 *
 * Parameters:
 * 	Flags - Initialization flag for CUDA.
 *
 * Returns:
 *  CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE 
 */
CUresult cuInit(unsigned int Flags)
{
	char buf[64];
	int minor = 0;
	int fd;

	/* mark initialized. */
	gdev_initialized = 1;

	/* the flag must be zero. */
	if (Flags != 0)
		return CUDA_ERROR_INVALID_VALUE;

	for (;;) {
		sprintf(buf, "/dev/gdev%d", minor);
		if ((fd = open(buf, O_RDWR, 0)) >= 0)
			close(fd);
		else
			break;
		minor++;
	}

	if (!minor)
		return CUDA_ERROR_INVALID_DEVICE;

	gdev_device_count = minor;

	return CUDA_SUCCESS;
}
