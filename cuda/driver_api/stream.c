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
#include "gdev_api.h"
#include "gdev_list.h"

/**
 * Creates a stream and returns a handle in phStream. Flags is required to be 0.
 *
 * Parameters:
 * phStream - Returned newly created stream
 * Flags - Parameters for stream creation (must be 0)
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, 
 * CUDA_ERROR_OUT_OF_MEMORY 
 */
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
	CUresult res;
	struct CUctx_st *ctx;
	struct CUstream_st *stream;
	Ghandle handle;
	int minor;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;

	if (!phStream || Flags != 0)
		return CUDA_ERROR_INVALID_VALUE;

	/* allocate a new stream. */
	stream = (CUstream)MALLOC(sizeof(*stream));
	if (!stream)
		return CUDA_ERROR_OUT_OF_MEMORY;
	
	/* create another channel for the stream. */
	minor = ctx->minor;
	if (!(handle = gopen(minor))) {
		res = CUDA_ERROR_UNKNOWN;
		goto fail_gopen;
	}

	stream->gdev_handle = handle;
	stream->ctx = ctx;
	gdev_list_init(&stream->sync_list, NULL);	
	gdev_list_init(&stream->event_list, NULL);	
	stream->wait = 0;

	*phStream = stream;

	return CUDA_SUCCESS;

fail_gopen:
	return res;
}

/**
 * Destroys the stream specified by hStream.
 *
 * In the case that the device is still doing work in the stream hStream when 
 * cuStreamDestroy() is called, the function will return immediately and the 
 * resources associated with hStream will be released automatically once the 
 * device has completed all work in hStream.
 *
 * Parameters:
 * hStream - Stream to destroy
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE 
 */ 
CUresult cuStreamDestroy(CUstream hStream)
{
	CUresult res;
	struct CUctx_st *ctx;
	struct CUstream_st *stream = hStream;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;
	if (ctx != stream->ctx)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (!stream)
		return CUDA_ERROR_INVALID_VALUE;

	/* synchronize with the stream before destroying it. */
	cuStreamSynchronize(stream);

	if (gclose(stream->gdev_handle))
		return CUDA_ERROR_UNKNOWN;

	FREE(stream);

	return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream)
{
	GDEV_PRINT("cuStreamQuery: Not Implemented Yet\n");
	return CUDA_SUCCESS;
}

/**
 * Waits until the device has completed all operations in the stream specified 
 *  by hStream. If the context was created with the CU_CTX_SCHED_BLOCKING_SYNC 
 * flag, the CPU thread will block until the stream is finished with all of its
 * tasks.
 *
 * Parameters:
 * hStream - Stream to wait for
 *
 * Returns:
 * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, 
 * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE 
 */
CUresult cuStreamSynchronize(CUstream hStream)
{
	CUresult res;
	struct CUctx_st *ctx;
	Ghandle handle;
	struct CUstream_st *stream = hStream;
	struct gdev_cuda_fence *f;
	struct gdev_list *p;
	TIME_T time;
	struct CUevent_st *e;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hStream)
		return CUDA_ERROR_INVALID_HANDLE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;
	if (ctx != stream->ctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (gdev_list_empty(&stream->sync_list))
		return CUDA_SUCCESS;

	handle = stream->gdev_handle;

	/* synchronize with all stream's tasks. */
	gdev_list_for_each(f, &stream->sync_list, list_entry) {
		/* if timeout is required, specify gdev_time value instead of NULL. */
		if (gsync(handle, f->id, NULL))
			return CUDA_ERROR_UNKNOWN;
	}

	/* complete event */
	GETTIME(&time);
	while ((p = gdev_list_head(&stream->event_list))) {
		gdev_list_del(p);
		e = gdev_list_container(p);
		e->time = time;
		e->record = 0;
		e->complete = 1;
	}

	/* remove all lists. */
	while ((p = gdev_list_head(&stream->sync_list))) {
		gdev_list_del(p);
		f = gdev_list_container(p);
		if (f->addr_ref)
			gunref(handle, f->addr_ref);
		FREE(f);
	}

	return CUDA_SUCCESS;
}

/**
 * Makes all future work submitted to hStream wait until hEvent reports
 * completion before beginning execution. This synchronization will be
 * performed efficiently on the device. The event hEvent may be from a
 * different context than hStream, in which case this function will perform
 * cross-device synchronization.
 *
 * The stream hStream will wait only for the completion of the most recent
 * host call to cuEventRecord() on hEvent. Once this call has returned,
 * any functions (including cuEventRecord() and cuEventDestroy()) may be
 * called on hEvent again, and subsequent calls will not have any effect
 * on hStream.
 *
 * If hStream is 0 (the NULL stream) any future work submitted in any
 * stream will wait for hEvent to complete before beginning execution.
 * This effectively creates a barrier for all future work submitted to
 * the context.
 *
 * If cuEventRecord() has not been called on hEvent, this call acts as
 * if the record has already completed, and so is a functional no-op.
 *
 * Parameters:
 *     hStream 	- Stream to wait
 *     hEvent 	- Event to wait on (may not be NULL)
 *     Flags 	- Parameters for the operation (must be 0)
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuStreamCreate, cuEventRecord, cuStreamQuery, cuStreamSynchronize,
 *     cuStreamDestroy 
 */
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hStream)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!hEvent)
		return CUDA_ERROR_INVALID_HANDLE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return res;
	if (ctx != hStream->ctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	hStream->wait = 1;

	if (hEvent->ctx == ctx) {
		res = cuEventSynchronize(hEvent);
	} else {
		while (hEvent->record)
			YIELD();
		res = CUDA_SUCCESS;
	}

	hStream->wait = 0;

	return res;
}
