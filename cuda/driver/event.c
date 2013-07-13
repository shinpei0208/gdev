/*
 * Copyright (C) 2011 Shinpei Kato
 * Copyright (C) 2013 AXE, Inc.
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
 * Creates an event *phEvent with the flags specified via Flags.
 * Valid flags include:
 *
 *     CU_EVENT_DEFAULT: Default event creation flag.
 *     CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use
 *                             blocking synchronization. A CPU thread that uses
 *                             cuEventSynchronize() to wait on an event created
 *                             with this flag will block until the event has
 *                             actually been recorded.
 *     CU_EVENT_DISABLE_TIMING: Specifies that the created event does not need
 *                              to record timing data. Events created with this
 *                              flag specified and the CU_EVENT_BLOCKING_SYNC
 *                              flag not specified will provide the best
 *                              performance when used with cuStreamWaitEvent()
 *                              and cuEventQuery().
 *
 * Parameters:
 *     phEvent 	- Returns newly created event
 *     Flags 	- Event creation flags
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
 *     CUDA_ERROR_OUT_OF_MEMORY 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuEventRecord, cuEventQuery, cuEventSynchronize, cuEventDestroy,
 *     cuEventElapsedTime 
 */
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
	CUresult res;
	struct CUevent_st *event;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!phEvent)
		return CUDA_ERROR_INVALID_VALUE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (!(event = (CUevent)MALLOC(sizeof(*event)))) {
		res = CUDA_ERROR_OUT_OF_MEMORY;
		goto fail_malloc_event;
	}

	event->record = 0;
	event->complete = 0;
	event->flags = Flags;
	event->ctx = ctx;
	event->stream = NULL;

	/* save the current context to the stack, if necessary. */
	gdev_list_init(&event->list_entry, event);

	*phEvent = event;

	return CUDA_SUCCESS;

fail_malloc_event:
	return res;
}

/**
 * Destroys the event specified by hEvent.
 *
 * In case hEvent has been recorded but has not yet been completed when
 * cuEventDestroy() is called, the function will return immediately and
 * the resources associated with hEvent will be released automatically
 * once the device has completed hEvent.
 *
 * Parameters:
 *     hEvent 	- Event to destroy
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuEventCreate, cuEventRecord, cuEventQuery, cuEventSynchronize,
 *     cuEventElapsedTime 
 */
CUresult cuEventDestroy(CUevent hEvent)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hEvent)
		return CUDA_ERROR_INVALID_HANDLE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (hEvent->ctx != ctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (hEvent->record)
		gdev_list_del(&hEvent->list_entry);

	FREE(hEvent);

	return CUDA_SUCCESS;
}

/**
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle).
 * This happens because the cuEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could
 * execute in between the two measured events, thus altering the timing in
 * a significant way.
 *
 * If cuEventRecord() has not been called on either event then
 * CUDA_ERROR_INVALID_HANDLE is returned. If cuEventRecord() has been called
 * on both events but one or both of them has not yet been completed (that is,
 * cuEventQuery() would return CUDA_ERROR_NOT_READY on at least one of the
 * events), CUDA_ERROR_NOT_READY is returned. If either event was created
 * with the CU_EVENT_DISABLE_TIMING flag, then this function will return
 * CUDA_ERROR_INVALID_HANDLE.
 *
 * Parameters:
 *     pMilliseconds 	- Time between hStart and hEnd in ms
 *     hStart 	- Starting event
 *     hEnd 	- Ending event
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
 *     CUDA_ERROR_NOT_READY 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuEventCreate, cuEventRecord, cuEventQuery, cuEventSynchronize,
 *     cuEventDestroy 
 */
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
	TIME_T elapsed;
	long long round;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hStart)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!hEnd)
		return CUDA_ERROR_INVALID_HANDLE;

	if (!hStart->complete)
		return CUDA_ERROR_NOT_READY;
	if (!hEnd->complete)
		return CUDA_ERROR_NOT_READY;

#ifdef __KERNEL__
	elapsed.tv_sec = hEnd->time.tv_sec - hStart->time.tv_sec;
	elapsed.tv_usec = hEnd->time.tv_usec - hStart->time.tv_usec;
	round = elapsed.tv_sec * 1000000 + elapsed.tv_usec;
	*pMilliseconds = (float)round / 1000.0;
#else
	elapsed.tv_sec = hEnd->time.tv_sec - hStart->time.tv_sec;
	elapsed.tv_nsec = hEnd->time.tv_nsec - hStart->time.tv_nsec;
	round = (elapsed.tv_sec * 1000000000 + elapsed.tv_nsec) / 500;
	*pMilliseconds = (float)round / 2000.0;
#endif

	return CUDA_SUCCESS;
}


/**
 * Query the status of all device work preceding the most recent call to
 * cuEventRecord() (in the appropriate compute streams, as specified by the
 * arguments to cuEventRecord()).
 *
 * If this work has successfully been completed by the device, or if
 * cuEventRecord() has not been called on hEvent, then CUDA_SUCCESS is returned.
 * If this work has not yet been completed by the device then
 * CUDA_ERROR_NOT_READY is returned.
 *
 * Parameters:
 *     hEvent 	- Event to query
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
 *     CUDA_ERROR_NOT_READY 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuEventCreate, cuEventRecord, cuEventSynchronize, cuEventDestroy,
 *     cuEventElapsedTime 
 */
CUresult cuEventQuery(CUevent hEvent)
{
	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hEvent)
		return CUDA_ERROR_INVALID_HANDLE;

	if (hEvent->record)
		return CUDA_ERROR_NOT_READY;

	return CUDA_SUCCESS;
}

/**
 * Records an event. If hStream is non-zero, the event is recorded after all
 * preceding operations in hStream have been completed; otherwise, it is
 * recorded after all preceding operations in the CUDA context have been
 * completed. Since operation is asynchronous, cuEventQuery and/or
 * cuEventSynchronize() must be used to determine when the event has actually
 * been recorded.
 *
 * If cuEventRecord() has previously been called on hEvent, then this call
 * will overwrite any existing state in hEvent. Any subsequent calls which
 * examine the status of hEvent will only examine the completion of this
 * most recent call to cuEventRecord().
 *
 * It is necessary that hEvent and hStream be created on the same context.
 *
 * Parameters:
 *     hEvent 	- Event to record
 *     hStream 	- Stream to record event for
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
 *     CUDA_ERROR_INVALID_VALUE 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuEventCreate, cuEventQuery, cuEventSynchronize, cuStreamWaitEvent,
 *     cuEventDestroy, cuEventElapsedTime 
 */
CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hEvent)
		return CUDA_ERROR_INVALID_HANDLE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (hEvent->ctx != ctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (hEvent->record)
		gdev_list_del(&hEvent->list_entry);

	if (hStream)
		gdev_list_add(&hEvent->list_entry, &hStream->event_list);
	else
		gdev_list_add(&hEvent->list_entry, &ctx->event_list);

	hEvent->stream = hStream;
	hEvent->record = 1;
	hEvent->complete = 0;

	return CUDA_SUCCESS;
}

/**
 * Wait until the completion of all device work preceding the most recent
 * call to cuEventRecord() (in the appropriate compute streams, as specified
 * by the arguments to cuEventRecord()).
 *
 * If cuEventRecord() has not been called on hEvent, CUDA_SUCCESS is returned
 * immediately.
 *
 * Waiting for an event that was created with the CU_EVENT_BLOCKING_SYNC flag
 * will cause the calling CPU thread to block until the event has been
 * completed by the device. If the CU_EVENT_BLOCKING_SYNC flag has not been
 * set, then the CPU thread will busy-wait until the event has been completed
 * by the device.
 *
 * Parameters:
 *     hEvent 	- Event to wait for
 *
 * Returns:
 *     CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *     CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE 
 *
 * Note:
 *     Note that this function may also return error codes from previous,
 *     asynchronous launches.
 *
 * See also:
 *     cuEventCreate, cuEventRecord, cuEventQuery, cuEventDestroy,
 *     cuEventElapsedTime 
 */
CUresult cuEventSynchronize(CUevent hEvent)
{
	CUresult res;
	struct CUctx_st *ctx;

	if (!gdev_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!hEvent)
		return CUDA_ERROR_INVALID_HANDLE;

	res = cuCtxGetCurrent(&ctx);
	if (res != CUDA_SUCCESS)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (hEvent->ctx != ctx)
		return CUDA_ERROR_INVALID_CONTEXT;

	if (hEvent->record) {
		if (hEvent->stream)
			res = cuStreamSynchronize(hEvent->stream);
		else
			res = cuCtxSynchronize();
		if (res != CUDA_SUCCESS)
			goto fail_sync;
	}

	return CUDA_SUCCESS;

fail_sync:
	return res;
}

