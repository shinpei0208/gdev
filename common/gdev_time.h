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
#include <linux/types.h>
#else
#include <sys/time.h>
#include <stdint.h>
#endif

#ifndef NULL
#define NULL 0
#endif

#define USEC_1SEC	1000000
#define USEC_1MSEC	1000
#define MSEC_1SEC	1000

/* compatible with struct timeval. */
struct gdev_time {
	uint64_t sec;
	uint64_t usec;
};

/* ret = current time. */
static inline
void gdev_time_stamp(struct gdev_time *ret)
{
	gettimeofday((struct timeval *) ret, NULL);	
}

/* ret = x + y. */
static inline 
void gdev_time_add(struct gdev_time *ret, struct gdev_time *x, struct gdev_time *y)
{
	ret->sec = x->sec + y->sec;
	ret->usec = x->usec + y->usec;
	if (ret->usec >= USEC_1SEC) {
		ret->sec++;
		ret->usec -= USEC_1SEC;
	}
}

/* ret = x - y. */
static inline 
void gdev_time_sub(struct gdev_time *ret, struct gdev_time *x, struct gdev_time *y)
{
	ret->sec = x->sec - y->sec;
	ret->usec = x->usec - y->usec;
	if (ret->usec < 0) {
		ret->sec--;
		ret->usec += USEC_1SEC;
	}
}

/* ret = x * I. */
static inline 
void gdev_time_mul(struct gdev_time *ret, struct gdev_time *x, int I)
{
	ret->sec = x->sec * I;
	ret->usec = x->usec * I;
	if (ret->usec >= USEC_1SEC) {
		unsigned long carry = ret->usec / USEC_1SEC;
		ret->sec += carry;
		ret->usec -= carry * USEC_1SEC;
	}
}

/* ret = x / I. */
static inline 
void gdev_time_div(struct gdev_time *ret, struct gdev_time *x, int I)
{
	ret->sec = x->sec / I;
	ret->usec = x->usec / I;
}

/* x >= y. */
static inline
int gdev_time_ge(struct gdev_time *x, struct gdev_time *y)
{
	return x->sec == y->sec ? 
		x->usec >= y->usec :
		x->sec >= y->sec;
}

/* x <= y. */
static inline 
int gdev_time_le(struct gdev_time *x, struct gdev_time *y)
{
	return x->sec == y->sec ? 
		x->usec <= y->usec :
		x->sec <= y->sec;
}

/* generate struct gdev_time from seconds. */
static inline 
void gdev_time_sec(struct gdev_time *ret, unsigned long sec)
{
	ret->sec = sec;
	ret->usec = 0;
}

/* generate struct gdev_time from milliseconds. */
static inline 
void gdev_time_ms(struct gdev_time *ret, unsigned long ms)
{
	unsigned long carry = ms / MSEC_1SEC;
	ret->sec = carry;
	ret->usec = (ms - carry * MSEC_1SEC) * USEC_1MSEC;
}

/* generate struct gdev_time from microseconds. */
static inline 
void gdev_time_us(struct gdev_time *ret, unsigned long us)
{
	ret->sec = 0;
	ret->usec = us;
}

/* clear the timeval values. */
static inline 
void gdev_time_clear(struct gdev_time *t)
{
	t->sec = t->usec = 0;
}

#endif
