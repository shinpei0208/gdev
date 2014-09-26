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

#ifndef __GDEV_CUDA_UTIL_H__
#define __GDEV_CUDA_UTIL_H__

#include "gdev_platform.h"
#include "gdev_platform_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/shm.h>

#include <elf.h>
#include <limits.h>
/* gpu only use 64bit */
#if 0
#define Elf_Ehdr Elf32_Ehdr
#define Elf_Shdr Elf32_Shdr
#define Elf_Phdr Elf32_Phdr
#define Elf_Sym	 Elf32_Sym
#else
#define Elf_Ehdr Elf64_Ehdr
#define Elf_Shdr Elf64_Shdr
#define Elf_Phdr Elf64_Phdr
#define Elf_Sym	 Elf64_Sym
#endif


/* a bit wild coding... */
static inline int __gdev_get_device_count(void)
{
	char fname[256] = "/proc/gdev/virtual_device_count";
	char buf[16];
	int minor = 0;
	FILE *fp;
	int shmid;
	int *shm;

	if (!(fp = fopen(fname, "r"))) {
		/* this is the case for non-gdev device drivers. */
		struct stat st;
		/* check for Linux open-source drivers first. */
		for (;;) {
			sprintf(fname, "/dev/dri/card%d", minor);
			if (stat(fname, &st))
				break;
			minor++;
		}
		if (minor)
			return minor;
		/* check for NVIDIA BLOB drivers next. */
		for (;;) {
			sprintf(fname, "/dev/nvidia%d", minor);
			if (stat(fname, &st))
				break;
			minor++;
		}
		/* check for Gdev user-scheduling */
		shmid = shmget( 0x600dde11, sizeof(int), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP );
		if(!shmid){
		    shm = (int *)shmat(shmid, NULL, 0);
		    minor = *shm;
		}
		return minor;
	}
	if (!fgets(buf, 16, fp))
		sprintf(buf, "0");
	fclose(fp);

	sscanf(buf, "%d", &minor);

	return minor;
}


#endif
