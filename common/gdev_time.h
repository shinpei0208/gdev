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

#ifndef __GDEV_TIME_H__
#define __GDEV_TIME_H__

#ifdef __KERNEL__
#define gettimeofday(x, y) do_gettimeofday(x)
#include <linux/time.h>
#else
#include <sys/time.h>
#endif

#define USEC_1SEC	1000000
#define USEC_1MSEC	1000
#define MSEC_1SEC	1000

typedef struct timeval gdev_time_t;

/* ret = current time. */
static inline
void gdev_time_stamp(gdev_time_t *ret)
{
	gettimeofday(ret, NULL);	
}

/* ret = x + y. */
static inline 
void gdev_time_add(gdev_time_t *ret, gdev_time_t *x, gdev_time_t *y)
{
	ret->tv_sec = x->tv_sec + y->tv_sec;
	ret->tv_usec = x->tv_usec + y->tv_usec;
	if (ret->tv_usec >= USEC_1SEC) {
		ret->tv_sec++;
		ret->tv_usec -= USEC_1SEC;
	}
}

/* ret = x - y. */
static inline 
void gdev_time_sub(gdev_time_t *ret, gdev_time_t *x, gdev_time_t *y)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += USEC_1SEC;
	}
}

/* ret = x * I. */
static inline 
void gdev_time_mul(gdev_time_t *ret, gdev_time_t *x, int I)
{
	ret->tv_sec = x->tv_sec * I;
	ret->tv_usec = x->tv_usec * I;
	if (ret->tv_usec >= USEC_1SEC) {
		unsigned long carry = ret->tv_usec / USEC_1SEC;
		ret->tv_sec += carry;
		ret->tv_usec -= carry * USEC_1SEC;
	}
}

/* ret = x / I. */
static inline 
void gdev_time_div(gdev_time_t *ret, gdev_time_t *x, int I)
{
	ret->tv_sec = x->tv_sec / I;
	ret->tv_usec = x->tv_usec / I;
}

/* x >= y. */
static inline
int gdev_time_ge(gdev_time_t *x, gdev_time_t *y)
{
	return x->tv_sec == y->tv_sec ? 
		x->tv_usec >= y->tv_usec :
		x->tv_sec >= y->tv_sec;
}

/* tvge: x <= y. */
static inline 
int gdev_time_le(gdev_time_t *x, gdev_time_t *y)
{
	return x->tv_sec == y->tv_sec ? 
		x->tv_usec <= y->tv_usec :
		x->tv_sec <= y->tv_sec;
}

/* generate gdev_time_t from seconds. */
static inline 
void gdev_time_sec(gdev_time_t *ret, unsigned long sec)
{
	ret->tv_sec = sec;
	ret->tv_usec = 0;
}

/* generate gdev_time_t from milliseconds. */
static inline 
void gdev_time_ms(gdev_time_t *ret, unsigned long ms)
{
	unsigned long carry = ms / MSEC_1SEC;
	ret->tv_sec = carry;
	ret->tv_usec = (ms - carry * MSEC_1SEC) * USEC_1MSEC;
}

/* generate gdev_time_t from microseconds. */
static inline 
void gdev_time_us(gdev_time_t *ret, unsigned long us)
{
	ret->tv_sec = 0;
	ret->tv_usec = us;
}

/* clear the timeval values. */
static inline 
void gdev_time_clear(gdev_time_t *tv)
{
	tv->tv_sec = tv->tv_usec = 0;
}

#endif
