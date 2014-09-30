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
	/* mark initialized. */
	gdev_initialized = 1;

	/* the flag must be zero. */
	if (Flags != 0)
		return CUDA_ERROR_INVALID_VALUE;

	if (gdevice_count(&gdev_device_count))
		return CUDA_ERROR_INVALID_DEVICE;

	gdev_list_init(&gdev_ctx_list, NULL);
	LOCK_INIT(&gdev_ctx_list_lock);

	return CUDA_SUCCESS;
}

/**
 * global variables. 
 */
int gdev_initialized = 0;
