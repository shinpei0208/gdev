/*
 * Copyright (C) Shinpei Kato
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
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
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

#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
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
	int neg;
};

/* ret = current time */
static inline void gdev_time_stamp(struct gdev_time *ret)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	ret->sec = tv.tv_sec;
	ret->usec = tv.tv_usec;
	ret->neg = 0;
}

/* generate struct gdev_time from seconds. */
static inline void gdev_time_sec(struct gdev_time *ret, unsigned long sec)
{
	ret->sec = sec;
	ret->usec = 0;
}

/* generate struct gdev_time from milliseconds. */
static inline void gdev_time_ms(struct gdev_time *ret, unsigned long ms)
{
	unsigned long carry = ms / MSEC_1SEC;
	ret->sec = carry;
	ret->usec = (ms - carry * MSEC_1SEC) * USEC_1MSEC;
	ret->neg = 0;
}

/* generate struct gdev_time from microseconds. */
static inline void gdev_time_us(struct gdev_time *ret, unsigned long us)
{
	ret->sec = 0;
	ret->usec = us;
	ret->neg = 0;
}

/* transform from struct gdev_time to seconds. */
static inline unsigned long gdev_time_to_sec(struct gdev_time *p)
{
	return (p->sec * USEC_1SEC + p->usec) / USEC_1SEC;
}

/* transform from struct gdev_time to milliseconds. */
static inline unsigned long gdev_time_to_ms(struct gdev_time *p)
{
	return (p->sec * USEC_1SEC + p->usec) / USEC_1MSEC;
}

/* transform from struct gdev_time to microseconds. */
static inline unsigned long gdev_time_to_us(struct gdev_time *p)
{
	return (p->sec * USEC_1SEC + p->usec);
}

/* clear the timeval values. */
static inline void gdev_time_clear(struct gdev_time *t)
{
	t->sec = t->usec = t->neg = 0;
}


/* x == y */
static inline int gdev_time_eq(struct gdev_time *x, struct gdev_time *y)
{
	return (x->sec == y->sec) && (x->usec == y->usec);
}

/* p == 0 */
static inline int gdev_time_eqz(struct gdev_time *p)
{
	return (p->sec == 0) && (p->usec == 0);
}

/* x > y */
static inline int gdev_time_gt(struct gdev_time *x, struct gdev_time *y)
{
	if (!x->neg && y->neg)
		return true;
	else if (x->neg && !y->neg)
		return false;
	else if (x->neg && y->neg)
		return (x->sec == y->sec) ? (x->usec < y->usec) : (x->sec < y->sec);
	else
		return (x->sec == y->sec) ? (x->usec > y->usec) : (x->sec > y->sec);
}

/* p > 0 */
static inline int gdev_time_gtz(struct gdev_time *p)
{
	return (!p->neg) && ((p->sec > 0) || (p->usec > 0));
}

/* x >= y */
static inline int gdev_time_ge(struct gdev_time *x, struct gdev_time *y)
{
	if (gdev_time_eq(x, y))
		return true;
	else
		return gdev_time_gt(x, y);
}

/* p >= 0 */
static inline int gdev_time_gez(struct gdev_time *p)
{
	return gdev_time_gtz(p) || gdev_time_eqz(p);
}

/* x < y */
static inline int gdev_time_lt(struct gdev_time *x, struct gdev_time *y)
{
	if (!x->neg && y->neg)
		return false;
	else if (x->neg && !y->neg)
		return true;
	else if (x->neg && y->neg)
		return (x->sec == y->sec) ? (x->usec > y->usec) : (x->sec > y->sec);
	else
		return (x->sec == y->sec) ? (x->usec < y->usec) : (x->sec < y->sec);
}

/* p < 0 */
static inline int gdev_time_ltz(struct gdev_time *p)
{
	return p->neg;
}

/* x <= y */
static inline int gdev_time_le(struct gdev_time *x, struct gdev_time *y)
{
	if (gdev_time_eq(x, y))
		return true;
	else
		return gdev_time_lt(x, y);
}

/* p <= 0 */
static inline int gdev_time_lez(struct gdev_time *p)
{
	return gdev_time_ltz(p) || gdev_time_eqz(p);
}

/* ret = x + y (x and y must be positive) */
static inline void __gdev_time_add_pos(struct gdev_time *ret, struct gdev_time *x, struct gdev_time *y)
{
	ret->sec = x->sec + y->sec;
	ret->usec = x->usec + y->usec;
	if (ret->usec >= USEC_1SEC) {
		ret->sec++;
		ret->usec -= USEC_1SEC;
	}
}

/* ret = x - y (x and y must be positive) */
static inline void __gdev_time_sub_pos(struct gdev_time *ret, struct gdev_time *x, struct gdev_time *y)
{
	if (gdev_time_lt(x, y)) {
		struct gdev_time *tmp = x;
		x = y;
		y = tmp;
		ret->neg = 1;
	}
	else
		ret->neg = 0;
	ret->sec = x->sec - y->sec;
	ret->usec = x->usec - y->usec;
	if (ret->usec < 0) {
		ret->sec--;
		ret->usec += USEC_1SEC;
	}
}

/* ret = x + y. */
static inline void gdev_time_add(struct gdev_time *ret, struct gdev_time *x, struct gdev_time *y)
{
	if (ret != x && ret != y)
		gdev_time_clear(ret);

	if (!x->neg && y->neg) { /* x - y */
		y->neg = 0;
		__gdev_time_sub_pos(ret, x, y);
		y->neg = 1;
	}
	else if (x->neg && !y->neg) { /* y - x */
		x->neg = 0;
		__gdev_time_sub_pos(ret, y, x);
		x->neg = 1;
	}
	else if (x->neg && y->neg) { /* - (x + y) */
		x->neg = y-> neg = 0;
		__gdev_time_add_pos(ret, x, y);
		ret->neg = 1;
		x->neg = y-> neg = 1;
	}
	else { /* x + y */
		__gdev_time_add_pos(ret, x, y);
	}
}

/* ret = x - y. */
static inline void gdev_time_sub(struct gdev_time *ret, struct gdev_time *x, struct gdev_time *y)
{
	if (ret != x && ret != y)
		gdev_time_clear(ret);

	if (!x->neg && y->neg) { /* x + y */
		y->neg = 0;
		__gdev_time_add_pos(ret, x, y);
		y->neg = 1;
	}
	else if (x->neg && !y->neg) { /* - (x + y) */
		x->neg = 0;
		__gdev_time_add_pos(ret, y, x);
		ret->neg = 1;
		x->neg = 1;
	}
	else if (x->neg && y->neg) { /* y - x */
		x->neg = y-> neg = 0;
		__gdev_time_sub_pos(ret, y, x);
		x->neg = y-> neg = 1;
	}
	else { /* x - y */
		__gdev_time_sub_pos(ret, x, y);
	}
}

/* ret = x * I. */
static inline void gdev_time_mul(struct gdev_time *ret, struct gdev_time *x, int I)
{
	if (ret != x)
		gdev_time_clear(ret);

	ret->sec = x->sec * I;
	ret->usec = x->usec * I;
	if (ret->usec >= USEC_1SEC) {
		unsigned long carry = ret->usec / USEC_1SEC;
		ret->sec += carry;
		ret->usec -= carry * USEC_1SEC;
	}
}

/* ret = x / I. */
static inline void gdev_time_div(struct gdev_time *ret, struct gdev_time *x, int I)
{
	if (ret != x)
		gdev_time_clear(ret);

	ret->sec = x->sec / I;
	ret->usec = x->usec / I;
}

#endif
