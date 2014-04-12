/*
 * Copyright (C)  Yusuke FUJII <emb.ubi.yukke@gmail.com>
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
#include "gdev_usched_monitor.h"

void __gdev_lock_init(struct gdev_lock *p)
{
    semid_array[semno] = p->semid = semget(GDEV_SEM_KEYS(semno), 1,
	    IPC_CREAT | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    semctl(p->semid, 0, SETVAL, 1);
    semno++;
    return;
}

void __gdev_init_device(struct gdev_device *gdev, int id)
{
    gdev->id = id;
    gdev->users = 0;
    gdev->accessed = 0;
    gdev->blocked = 0;
    gdev->mem_size = 0;
    gdev->mem_used = 0;
    gdev->dma_mem_size = 0;
    gdev->dma_mem_used = 0;
    gdev->chipset = 0;
    gdev->com_bw = gdev_bw_set[id];
    gdev->mem_bw = gdev_bw_set[id];
    gdev->mem_sh = 100;
    gdev->com_bw_used = 0;
    gdev->mem_bw_used = 0;
    gdev->period = 0;
    gdev->com_time = 0;
    gdev->mem_time = 0;
    gdev->swap = NULL;
    gdev->sched_com_thread = NULL;
    gdev->sched_mem_thread = NULL;
    gdev->credit_com_thread = NULL;
    gdev->credit_mem_thread = NULL;
    gdev->current_com = NULL;
    gdev->current_mem = NULL;
    gdev->parent = NULL;//(void *)(gdev - id);
    gdev->priv = NULL;
    gdev_time_us(&gdev->credit_com, 0);
    gdev_time_us(&gdev->credit_mem, 0);
    gdev_list_init(&gdev->sched_com_list, NULL);
    gdev_list_init(&gdev->sched_mem_list, NULL);
    gdev_list_init(&gdev->vas_list, NULL);
    gdev_list_init(&gdev->shm_list, NULL);
    __gdev_lock_init(&gdev->sched_com_lock);
    __gdev_lock_init(&gdev->sched_mem_lock);
    __gdev_lock_init(&gdev->vas_lock);
    __gdev_lock_init(&gdev->global_lock);
    gdev_mutex_init(&gdev->shm_mutex);
}

void __gdev_init_sched_entity(struct gdev_sched_entity *se, int id)
{
    se->gdev=NULL;
    se->task=NULL;
    se->ctx=NULL;
    se->prio=0;
    se->rt_prio = GDEV_PRIO_DEFAULT;
}


void __gdev_init_vas(struct gdev_vas *vas){
    __gdev_lock_init(&vas->lock);
}
void gdev_mutex_init(struct gdev_mutex *p){
    __gdev_lock_init((struct gdev_lock*)p);
    return;
}

int __attach_shm(int shmid, void **ptr, int size, int offset)
{

    if (shmid == -1){
	GDEV_PRINT("Failed get shared memory for user-space scheduling\n");
	return false;
    }

    *ptr =(void *)((unsigned long long)shmat(shmid, NULL, 0)+(unsigned long long)offset);
    if (*ptr == (void *) -1)
	return false;
    return true;
}

int __alloc_shm(int device_size)
{
    int __shmid;

    __shmid = shmget(GDEV_SHM_KEYS(1),
	    sizeof(struct gdev_device) *device_size+ 
	    sizeof(struct gdev_vas) * GDEV_CONTEXT_MAX_COUNT +
	    sizeof(struct gdev_sched_entity)*GDEV_CONTEXT_MAX_COUNT +
	    sizeof(struct gdev_mem) * GDEV_CONTEXT_MAX_COUNT * 10,
	    IPC_CREAT | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (__shmid == -1){
	GDEV_PRINT("Failed get shared memory for user-space scheduling\n");
	return false;
    }
    return __shmid;
}
int init_gdev_monitor()
{
    int attach_offset=0;
    int i;

    /* allocate Gdev all objects(shared memory).  */
    shmid = __alloc_shm(GDEV_DEVICE_MAX_COUNT);

    /* attach gdev device objects */
    __attach_shm(shmid, (void**)&gdevs, 0, attach_offset);
    attach_offset += sizeof(*gdevs) * GDEV_DEVICE_MAX_COUNT;
    for (i = 0; i < gdev_count + (*_gdev_vcount); i++)
	__gdev_init_device(&(gdevs[i]), i);

    GDEV_PRINT("Found %d physical device(s).\n", gdev_count);
    GDEV_PRINT("Configured %d virtual device(s).\n", *_gdev_vcount);

    __attach_shm(shmid, (void**)&vas, 0, attach_offset);
    attach_offset += sizeof(struct gdev_vas)* GDEV_CONTEXT_MAX_COUNT;
    for (i = 0; i< gdev_count + (*_gdev_vcount); i++)
	__gdev_init_vas(&(vas[i]));

    GDEV_PRINT("Prepare %d VAS entities(s).\n", GDEV_CONTEXT_MAX_COUNT);

    __attach_shm(shmid, (void **)&se, GDEV_CONTEXT_MAX_COUNT, attach_offset);
    attach_offset += sizeof(struct gdev_sched_entity)* GDEV_CONTEXT_MAX_COUNT;
    for (i = 0; i < GDEV_CONTEXT_MAX_COUNT; i++)
	__gdev_init_sched_entity(&(se[i]), i);

    GDEV_PRINT("Prepared %d scheduling entities.\n",GDEV_CONTEXT_MAX_COUNT);

    __attach_shm(shmid, (void **)&mem, 256, attach_offset);
    attach_offset += sizeof(struct gdev_mem) * 256;

    GDEV_PRINT("Prepared %d memory entities.\n",256);

    /*
     *
     */


    msgid = msgget(GDEV_MSG_KEY, IPC_CREAT| S_IRUSR| S_IWUSR | S_IRGRP | S_IWGRP);
    if (msgid==-1) {
	printf("Failed to initialize gdev message\n");
	exit(1);
    }
    return 1;
}

