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

#ifndef __GDEV_CUDA_UTIL_H__
#define __GDEV_CUDA_UTIL_H__

#include <linux/elf.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/vmalloc.h>
typedef struct file file_t;
#ifdef CONFIG_64BIT
#define Elf_Ehdr Elf64_Ehdr
#define Elf_Shdr Elf64_Shdr
#define Elf_Phdr Elf64_Phdr
#define Elf_Sym	 Elf64_Sym
#else
#define Elf_Ehdr Elf32_Ehdr
#define Elf_Shdr Elf32_Shdr
#define Elf_Phdr Elf32_Phdr
#define Elf_Sym	 Elf32_Sym
#endif
static inline file_t *FOPEN(const char *fname)
{
	file_t *fp = filp_open(fname, O_RDONLY, 0);
	return IS_ERR(fp) ? NULL: fp;
}
#define FSEEK(fp, offset, whence) generic_file_llseek(fp, 0, whence)
#define FTELL(fp) (fp)->f_pos
#define FREAD(ptr, size, fp) kernel_read(fp, 0, ptr, size)
#define FCLOSE(fp) filp_close(fp, NULL)
#define MALLOC(x) vmalloc(x)
#define FREE(x) vfree(x)
#define GDEV_PRINT(fmt, arg...) printk("[gdev] " fmt, ##arg)

/**
 * Gdev getinfo functions (exported to kernel modules).
 * the same information can be found in /proc/gdev/ for user-space.
 */
extern int gdev_getinfo_device_count(void);

static inline int __gdev_get_device_count(void)
{
	return gdev_getinfo_device_count();
}

#endif
