/*
* Copyright (C) Shinpei Kato
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

#ifndef __GDEV_LIB_H__
#define __GDEV_LIB_H__

#include <errno.h> /* ENOMEN, etc. */
#include <sched.h> /* sched_yield, etc. */
#include <stdint.h> /* uint32_t, etc.*/
#include <stdio.h> /* printf, etc. */
#include <stdlib.h> /* malloc/free, etc. */
#include <string.h> /* memcpy, etc. */
#include <sys/sem.h> /* struct sembuf */
#include <sys/msg.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>

#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
#endif

#define GDEV_SHM_KEY 0xdeadbabe
#define GDEV_SHM_KEYS(x) (0xdead0000|x)
#define GDEV_SHM_SE_KEY GDEV_SHM_KEYS(0x100)
#define GDEV_SEM_KEY 0x600dbeef
#define GDEV_SEM_KEYS(x) (0x600d0000|x)
#define GDEV_MSG_KEY 0xdeadcafe

#define GDEV_NR_TASKS 64
#define GDEV_SZ_MSG sizeof(struct gdev_msg_struct)
#define GDEV_SZ_OPS sizeof(struct gdev_msgops_struct)

struct gdev_lock {
	int semid;
	struct sembuf sembuf;
};

struct gdev_mutex {
	int semid;
	struct sembuf sembuf;
};

struct gdev_msg_struct {
    long mtype;
    char mtext;
};

extern int gdev_shm_initialized;
extern struct gdev_time now,last,elapse,interval;

#endif
