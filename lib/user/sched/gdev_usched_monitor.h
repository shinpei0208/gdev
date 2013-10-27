#ifndef __GDEV_USCHED_MONITOR__
#define __GDEV_USCHED_MONITOR__

#include "gdev_lib.h"
#include "gdev_sched.h"
#include "gdev_nvidia.h"
#include <pthread.h>

#define GDEV_DEVICE_MAX_COUNT 32
#define VDEVICE_MAX_COUNT 16
#define GDEV_PERIOD_DEFAULT 30000 /* microseconds */

extern struct gdev_device *phys;
extern struct gdev_device *gdevs;
extern struct gdev_sched_entity *se;
extern struct gdev_mem *mem;
extern struct gdev_vas *vas;
extern int shmid;
extern int semid;
extern int msgid;
extern int gdev_count;
extern int *_gdev_vcount;
extern int semno;
extern int semid_array[];
extern int gdev_bw_set[];

void gdev_mutex_init(struct gdev_mutex *);
int init_gdev_monitor();

#endif 
