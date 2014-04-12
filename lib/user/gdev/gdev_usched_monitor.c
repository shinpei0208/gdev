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


struct gdev_device *phys;
struct gdev_device *gdevs;
struct gdev_sched_entity *se;
struct gdev_mem *mem;
struct gdev_vas *vas;

int shmid;
int semid;
int msgid;
int gdev_count;
int *_gdev_vcount;
int semno = 0;

int semid_array[VDEVICE_MAX_COUNT+1];
int gdev_bw_set[VDEVICE_MAX_COUNT]={
    0,  /* phys   */
    25, /* vgpu0  */
    25,
    25,
    25,
};

void __gdev_lock(gdev_lock_t *p)
{
}
void __gdev_unlock(gdev_lock_t *p)
{
}


void gdev_vsched_band_replenish_compute(struct gdev_device *gdev)
{
    struct gdev_time credit, threshold;
    gdev_time_us(&credit, gdev->period * gdev->com_bw / 100);
    gdev_time_add(&gdev->credit_com, &gdev->credit_com, &credit);
    /* when the credit exceeds the threshold, all credits taken away. */
    gdev_time_us(&threshold, GDEV_CREDIT_INACTIVE_THRESHOLD);
    if (gdev_time_gt(&gdev->credit_com, &threshold))
	gdev_time_us(&gdev->credit_com, 0);
    /* when the credit exceeds the threshold in negative, even it. */
    threshold.neg = 1;
    if (gdev_time_lt(&gdev->credit_com, &threshold))
	gdev_time_us(&gdev->credit_com, 0);
}


void gdev_vsched_credit_replenish_compute(struct gdev_device *gdev)
{
    GDEV_PRINT("not implemented....\n");
    return;
}

void gdev_vsched_fifo_replenish_compute(struct gdev_device *gdev)
{
    GDEV_PRINT("not implemented....\n");
    return;
}

void gdev_vsched_null_replenish_compute(struct gdev_device *gdev)
{
    return;
}

#define GDEV_VSCHED_POLICY_BAND
//#define GDEV_VSCHED_POLICY_CREDIT
//#define GDEV_VSCHED_POLICY_FIFO
//#define GDEV_VSCHED_POLICY_NULL

#if defined(GDEV_VSCHED_POLICY_BAND)
void (*gdev_replenish_compute)(struct gdev_device *gdev) = &gdev_vsched_band_replenish_compute;
#elif defined(GDEV_VSCHED_POLICY_CREDIT)
void (*gdev_replenish_compute)(struct gdev_device *gdev) = &gdev_vsched_credit_replenish_compute;
#elif defined(GDEV_VSCHED_POLICY_FIFO)
void (*gdev_replenish_compute)(struct gdev_device *gdev) = &gdev_vsched_fifo_replenish_compute;
#elif defined(GDEV_VSCHED_POLICY_NULL)
void (*gdev_replenish_compute)(struct gdev_device *gdev) = &gdev_vsched_null_replenish_compute;
#endif

static void *__gdev_credit_com_thread(void *__offset)
{
    struct gdev_device *gdev = (struct gdev_device*)((unsigned long long)shmat(shmid, NULL, 0) + (unsigned long long)__offset);
    struct gdev_time now, last, elapse, interval;

    GDEV_PRINT("Gdev#%d compute reserve running\n", gdev->id);

    gdev_time_us(&interval, GDEV_UPDATE_INTERVAL);
    gdev_time_stamp(&last);

    while(1){
	gdev_replenish_compute(gdev);
	pthread_testcancel();
	usleep(GDEV_PERIOD_DEFAULT * 1000);    
	__gdev_lock(&gdev->sched_com_lock);
	gdev_time_stamp(&now);
	gdev_time_sub(&elapse, &now, &last);
	gdev->com_bw_used = gdev->com_time * 100 / gdev_time_to_us(&elapse);
	if (gdev->com_bw_used > 100)
	    gdev->com_bw_used = 100;
	if (gdev_time_ge(&elapse, &interval)) {
	    gdev->com_time = 0;
	    gdev_time_stamp(&last);
	}
	__gdev_unlock(&gdev->sched_com_lock);
    }
    return NULL;
}




static void __exit_gdev_monitor(void)
{
    int i;
    for (i = 0; i < semno; i++)
	semctl(semid_array[i], 0, IPC_RMID);

    shmctl(shmid, IPC_RMID, NULL);
    msgctl(msgid, IPC_RMID, NULL);
}

static void __kill_handler(int signum)
{
    /* unregister the KILL signal */
    struct sigaction sa_kill;

    memset(&sa_kill, 0, sizeof(sa_kill));
    sigemptyset(&sa_kill.sa_mask);
    sa_kill.sa_handler = SIG_DFL;
    sa_kill.sa_flags = 0;
    sigaction(SIGINT, &sa_kill, NULL);
    __exit_gdev_monitor();

    /* use kill() instead of exit(). */
    kill(0, SIGINT);
}

static void __display(void)
{
    static int all_count=0;
    int i;

    struct gdev_device *p;
    struct gdev_sched_entity *se_cu;

    printf("[%d]\t\tusers\tused/com_bw\tused/mem_bw\tcom_time\tmem_time\tparent\n",all_count++);
    printf("------------------------------------------------------------------------------\n");

    for(i = 0;i< gdev_count+(*_gdev_vcount); i++){
	p= &(gdevs[i]);
	if(i==((gdev_count-1)*(*_gdev_vcount)+(gdev_count-1))){
	    printf("[phys#%d]:\t%d\t%d/%d\t\t%d/%d\t\t%d\t\t%d\t\tNULL\n",
		    i,p->users,
		    p->com_bw_used,p->com_bw,
		    p->mem_bw_used,p->mem_bw,
		    p->com_time,p->mem_time);
	}else{
	    printf("[vgpu#%d]:\t%d\t%d/%d\t\t%d/%d\t\t%d\t\t%d\t\t0x%lx\n",
		    i-1,p->users,
		    p->com_bw_used,p->com_bw,
		    p->mem_bw_used,p->mem_bw,
		    p->com_time,p->mem_time,(long unsigned int)p->parent);
	}
    }
    for (i = 0; i < GDEV_CONTEXT_MAX_COUNT; i++){
	se_cu = &se[i];
	if(se_cu->list_entry_com.next!=0){
	    printf("se#%d!",i);
	}

    }
    printf("\n");
}


int main(int argc, char *argv[])
{
    int i;
    int __shmid_vcount;

    /*  register the KILL signal.  */
    struct sigaction sa_kill;
    memset(&sa_kill, 0, sizeof(sa_kill));
    sigemptyset(&sa_kill.sa_mask);
    sa_kill.sa_handler = __kill_handler;
    sa_kill.sa_flags = 0;
    sigaction(SIGINT, &sa_kill, NULL);

    __shmid_vcount = shmget(0x600dde11, sizeof(int), IPC_CREAT | S_IRUSR| S_IWUSR| S_IRGRP| S_IWGRP);
    _gdev_vcount = (int *)shmat(__shmid_vcount, NULL, 0);

    /* get the number of physical devices */
    gdev_count = 1; //fix this 
    /* get the number of virtual devices. */
    if (argc == 1){
	GDEV_PRINT("Use 2 virtual device (default)\n");
	*_gdev_vcount = 2; //fix this
    }else if (argc == 2){
	GDEV_PRINT("Use %d virtual device\n",atoi(argv[1]));
	*_gdev_vcount = atoi(argv[1]);
    }else{
	GDEV_PRINT("usage: gdev_usched_monitor \n");
	exit(1);
    }

    if (!init_gdev_monitor()){
	GDEV_PRINT("Initialize Error\n");
	return 0;
    }

    pthread_t thread[4];

    for ( i = 1; i<= *_gdev_vcount; i++){
	if (pthread_create(&thread[0], NULL, __gdev_credit_com_thread, (void*)((unsigned long long)&gdevs[i]-(unsigned long long)&gdevs[0]))!=0){
	    perror("pthread_create()\n");
	}
    }

    /* clean up zombie descriptors periodically  */
    while(1){
	//      __cleanup_zombie_descriptors();
	sleep(3);
	__display();
    }

    /* shouldn't reach here */
    __exit_gdev_monitor();
    return 0;
}

