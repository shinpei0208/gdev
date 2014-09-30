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

#include "gdev_autogen.h"
#include "gdev_device.h"
#include "gdev_lib.h"
#include "gdev_sched.h"
#include "gdev_util.h"

#ifdef GDEV_SCHED_DISABLED

int gdev_sched_create_scheduler(struct gdev_device *gdev)
{
        return 0;
}

void gdev_sched_destroy_scheduler(struct gdev_device *gdev)
{
}

void *gdev_sched_get_current_task(void)
{
        return NULL;
}

int gdev_sched_get_static_prio(void *task)
{
        return 0;
}

void gdev_sched_sleep(void)
{
}

int gdev_sched_wakeup(void *task)
{
        return 0;
}

void gdev_lock_init(struct gdev_lock *p)
{
}

void gdev_lock(struct gdev_lock *p)
{
}

void gdev_unlock(struct gdev_lock *p)
{
}

void gdev_lock_save(struct gdev_lock *p, unsigned long *pflags)
{
}

void gdev_unlock_restore(struct gdev_lock *p, unsigned long *pflags)
{
}

void gdev_lock_nested(struct gdev_lock *p)
{
}

void gdev_unlock_nested(struct gdev_lock *p)
{
}

void gdev_mutex_init(struct gdev_mutex *p)
{
}

void gdev_mutex_lock(struct gdev_mutex *p)
{
}

void gdev_mutex_unlock(struct gdev_mutex *p)
{
}

void* gdev_current_com_get(struct gdev_device *gdev)
{
        return !gdev->current_com? NULL:(void*)gdev->current_com;
}
void gdev_current_com_set(struct gdev_device *gdev,void *com)
{
        gdev->current_com = com;
}

void* gdev_compute_get(struct gdev_device *gdev)
{
        return gdev->compute;
}

void* gdev_priv_get(struct gdev_device *gdev)
{
        return gdev->priv;
}

struct gdev_mem *gdev_swap_get(struct gdev_device *gdev)
{
    	return gdev->swap;
}

struct gdev_sched_entity *gdev_sched_entity_alloc(int size){
        return (struct gdev_sched_entity*)malloc(sizeof(struct gdev_sched_entity));
}

struct gdev_device *gdev_phys_get(struct gdev_device *gdev)
{
        return !gdev?NULL:gdev->parent;
}

#else /* for User-Space Scheduling*/

int __msgid;
int gdev_shm_initialized=0;
int device_count=0;
void *attach_mem=NULL;
extern struct gdev_device *lgdev;
struct gdev_time now,last,elapse,interval;

static int __alloc_shm(int dev_size)
{
        int shmid;

        shmid = shmget(GDEV_SHM_KEYS(1),/*fix this: SHM_KEYS(x) value, */
	        sizeof(struct gdev_device) * dev_size +
	        sizeof(struct gdev_vas) * GDEV_CONTEXT_MAX_COUNT +
	        sizeof(struct gdev_sched_entity) * GDEV_CONTEXT_MAX_COUNT +
	        sizeof(struct gdev_mem) * GDEV_CONTEXT_MAX_COUNT * 10,
	        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
        if (shmid == -1){
	GDEV_PRINT("Failed get shared memory for user-space scheduling\nPlease start/restart gdev_usched_monitor\n");
	return false;
        }
        attach_mem = shmat(shmid, NULL, 0);
        return true;
}

static void *__attach_shms(int size,int attach_offset)
{
        if(!attach_mem)
	    __alloc_shm(size);
        return (void *)ADDR_ADD(attach_mem, attach_offset);
}

#define GDEV_DEVICE_MAX_COUNT 32
void *gdev_attach_shms_dev(int size)
{
        void *base;
        base = __attach_shms(GDEV_DEVICE_MAX_COUNT, 0);
	device_count = GDEV_DEVICE_MAX_COUNT;
        return base;
}

void *gdev_attach_shms_vas(int size)
{
        struct gdev_vas *base;
        int i;

        base = (struct gdev_vas *)__attach_shms(size/sizeof(struct gdev_device), 
		device_count * sizeof(struct gdev_device));
        for (i = 0; i < 128; i++){
	    if (!base->gdev)
		break;
	    base++;
	}
        return base;
}

void *gdev_attach_shms_se(int size)
{
        struct gdev_sched_entity *base;
        int i;

        base = __attach_shms(size/sizeof(struct gdev_device), 
		device_count * sizeof(struct gdev_device) +
		GDEV_CONTEXT_MAX_COUNT * sizeof(struct gdev_vas));
        for (i=0; i<128; i++){
	if (!base->gdev)
	        break;
	base++;
        }
        return base;
}

void *gdev_attach_shms_mem(int size)
{
        struct gdev_mem *base;
        int i;

        base = __attach_shms(size/sizeof(struct gdev_device),
		device_count * sizeof(struct gdev_device) + 
		GDEV_CONTEXT_MAX_COUNT * sizeof(struct gdev_vas) + 
		GDEV_CONTEXT_MAX_COUNT * sizeof(struct gdev_sched_entity));
	for ( i=0; i<256; i++){
	    if (!base->vas)
		break;
	    base++;
	}
        return base;

}

struct gdev_sched_entity *gdev_sched_entity_alloc(int size)
{
        return gdev_attach_shms_se(size);
}

static int __init_sem_lock(void)
{
        return true;
}

static int __init_msg_sched(void)
{
        __msgid = msgget(GDEV_MSG_KEY, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP);
        if (__msgid == -1)
		return false;

        return true;
}


int gdev_sched_create_scheduler(struct gdev_device *gdev)
{
        /* init the message queue for a scheduler. */
    	if (!__init_msg_sched()) {
		printf("Failed to initialize GDEV semaphore\n");
		exit(1);
		return false;
    	}
        gdev_shm_initialized=1;

        return true;
}


void gdev_sched_destroy_scheduler(struct gdev_device *gdev)
{
}

void *gdev_sched_get_current_task(void)
{
	if (!gdev_shm_initialized)
	        gdev_sched_create_scheduler(NULL);

        return (void*)(long long)getpid();
}

int gdev_sched_get_static_prio(void *task)
{
        return (int)getpriority(PRIO_PROCESS, (long long)task);
}
int gdev_sched_set_static_prio(void *task, int prio)
{
        return (int)setpriority(PRIO_PROCESS, (long long)task, prio);
}


void gdev_sched_sleep(void)
{
        int pid = getpid();
        struct gdev_msg_struct *msg = (struct gdev_msg_struct *)malloc(sizeof(*msg));

        msgrcv(__msgid, msg, GDEV_SZ_MSG, pid, 0);
}



int gdev_sched_wakeup(void *task)
{
        int pid=getpid();

	if ((long long)task == pid){
	    printf("Warning: task tried to wake up itself\n");
        }
        else {
	    struct gdev_msg_struct *msg;
	    msg = (struct gdev_msg_struct *)malloc(sizeof(*msg));
	    msg->mtype = (long long)task;
	    msg->mtext = (long long)task;
	    msgsnd(__msgid, msg, GDEV_SZ_MSG, 0);
	}
        return true;
}

void gdev_next_compute(struct gdev_device *gdev)
{
	if (gdev->users)
		gdev_select_next_compute(gdev);
}


void gdev_lock_init(struct gdev_lock *p)
{
        return;
}

void gdev_lock(struct gdev_lock *p)
{
        struct sembuf *sem = &p->sembuf;
        sem->sem_num = 0;
        sem->sem_op = -1;
        sem->sem_flg = SEM_UNDO;
        semop(p->semid, sem, 1);

}

void gdev_unlock(struct gdev_lock *p)
{
        struct sembuf *sem = &p->sembuf;
        sem->sem_num = 0;
        sem->sem_op = 1;
        sem->sem_flg = SEM_UNDO;
        semop(p->semid, sem, 1);
}

void gdev_lock_save(struct gdev_lock *p, unsigned long *pflags)
{
}

void gdev_unlock_restore(struct gdev_lock *p, unsigned long *pflags)
{
}

void gdev_lock_nested(struct gdev_lock *p)
{
}

void gdev_unlock_nested(struct gdev_lock *p)
{
}

void gdev_mutex_init(struct gdev_mutex *p)
{
}

void gdev_mutex_lock(struct gdev_mutex *p)
{
}

void gdev_mutex_unlock(struct gdev_mutex *p)
{
}


void gdev_enqueue(struct gdev_sched_entity *se)
{
}

void gdev_dequeue(struct gdev_sched_entity *se)
{
}

struct gdev_device *gdev_phys_get(struct gdev_device *gdev)
{
        return !gdev?NULL:(struct gdev_device*)ADDR_SUB(gdev,gdev->parent);
}

struct gdev_mem *gdev_swap_get(struct gdev_device *gdev)
{
 	return lgdev->swap;
}

void *gdev_current_com_get(struct gdev_device *gdev)
{
 	return (gdev->current_com==NULL)? NULL:(void*)ADDR_SUB(gdev, gdev->current_com);
}

void gdev_current_com_set(struct gdev_device *gdev,void *com)
{
	if(com!=NULL){
		gdev->current_com = (void *)ADDR_SUB(gdev,com);
    	}else{
		gdev->current_com = NULL;
    	}
}

void *gdev_compute_get(struct gdev_device *gdev)
{
        return lgdev->compute;
}

void *gdev_priv_get(struct gdev_device *gdev)
{
        return lgdev->priv;
}

#endif

int gdev_getinfo_device_count(void)
{
	char fname[256] = { 0 };
	int minor = 0;
	int shmid;
	int *shm;

#if defined(GDEV_DRIVER_BARRA)
	return 1;
#endif

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
