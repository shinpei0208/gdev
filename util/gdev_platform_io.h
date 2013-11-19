/*
 * Copyright (C) Yusuke Suzuki
 *
 * Keio University
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

#ifndef __GDEV_PLATFORM_IO_H__
#define __GDEV_PLATFORM_IO_H__

#ifdef __KERNEL__ /* OS functions */
#include <linux/fs.h>
#else /* user-space functions */
#include <stdio.h>
#endif

#ifdef __KERNEL__ /* OS functions related to File IO */
typedef struct file file_t;
static inline file_t *FOPEN(const char *fname)
{
	file_t *fp = filp_open(fname, O_RDONLY, 0);
	return IS_ERR(fp) ? NULL: fp;
}
#define FSEEK(fp, offset, whence) generic_file_llseek(fp, 0, whence)
#define FTELL(fp) (fp)->f_pos
#define FREAD(ptr, size, fp) kernel_read(fp, 0, ptr, size)
#define FCLOSE(fp) filp_close(fp, NULL)
#else /* user-space functions */
typedef FILE file_t;
#define FOPEN(fname) fopen(fname, "rb")
#define FSEEK(fp, offset, whence) fseek(fp, 0, whence)
#define FTELL(fp) ftell(fp)
#define FREAD(ptr, size, fp) fread(ptr, size, 1, fp)
#define FCLOSE(fp) fclose(fp)
#endif

#endif  /* __GDEV_PLATFORM_IO_H__ */
