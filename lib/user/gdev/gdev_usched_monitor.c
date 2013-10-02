/*
 * Yusuke FUJII
 *
 */
#include "gdev_lib.h"
#include "gdev_sched.h"
#include "gdev_nvidia.h"

#define GDEV_DEVICE_MAX_COUNT 32
#define VDEVICE_MAX_COUNT 16

struct gdev_device *phys;
struct gdev_device *__gdevs;
struct gdev_sched_entity *__se;
struct gdev_mem *__mem;
struct gdev_vas *__vas;

int __shmid;
int __semid;
int __msgid;
int gdev_count;
int gdev_vcount;
static int semno = 0;

int semid_array[VDEVICE_MAX_COUNT+1];
int gdev_bw_set[VDEVICE_MAX_COUNT]={
    0,  /* phys   */
    25, /* vgpu0  */
    25,
    25,
    25,
};

void gdev_lock_init(struct gdev_lock *p)
{
    semid_array[semno] = p->semid = semget(GDEV_SEM_KEYS(semno), 1,
	    IPC_CREAT | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    semctl(p->semid, 0, SETVAL, 1);
    semno++;
    return;
}
void gdev_mutex_init(struct gdev_mutex *p){
    gdev_lock_init((struct gdev_lock*)p);
    return;
}


static void __gdev_init_device(struct gdev_device *gdev, int id)
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
    gdev_lock_init(&gdev->sched_com_lock);
    gdev_lock_init(&gdev->sched_mem_lock);
    gdev_lock_init(&gdev->vas_lock);
    gdev_lock_init(&gdev->global_lock);
    gdev_mutex_init(&gdev->shm_mutex);
}

static void __gdev_init_sched_entity(struct gdev_sched_entity *se, int id)
{
    se->gdev=NULL;
    se->task=NULL;
    se->ctx=NULL;
    se->prio=0;
    se->rt_prio = GDEV_PRIO_DEFAULT;
}

static void __gdev_init_vas(struct gdev_vas *vas){
    gdev_lock_init(&vas->lock);
}



static int __alloc_shm(int device_size)
{
    int shmid;

    shmid = shmget(GDEV_SHM_KEYS(1),
	    sizeof(struct gdev_device) *device_size+ 
	    sizeof(struct gdev_vas) * GDEV_CONTEXT_MAX_COUNT +
	    sizeof(struct gdev_sched_entity)*GDEV_CONTEXT_MAX_COUNT +
	    sizeof(struct gdev_mem) * GDEV_CONTEXT_MAX_COUNT * 10,
	    IPC_CREAT | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (shmid == -1){
	GDEV_PRINT("Failed get shared memory for user-space scheduling\n");
	return false;
    }
    return shmid;
}

static int __attach_shm(int shmid, void **ptr, int size, int offset)
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

/*
static int __init_sem_lock(s)
{
    __semid = semget(GDEV_SEM_KEY, 1,
	    IPC_CREAT | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (__semid == -1)
	return false;

    if (semctl(__semid, 0, SETVAL, 1) != 0)
	return false;
    return true;
}
*/
static int __init_msg_sched(void)
{
    __msgid = msgget(GDEV_MSG_KEY, IPC_CREAT| S_IRUSR| S_IWUSR | S_IRGRP | S_IWGRP);
    if (__msgid == -1)
	return false;
    return true;
}

static void __init_gdev_monitor(void)
{
    /* init the semaphore for enqueue/dequeue lock. */
    /*
       if (!__init_sem_lock()) {
       printf("Failed to initialize Gdev semaphore\n");
       exit(1);
       }
       */
    /* init the message queue for a scheduler */
    if (!__init_msg_sched()) {
	printf("Failed to initialize gdev message\n");
	exit(1);
    }

}

static void __exit_gdev_monitor(void)
{
    int i;
    for (i = 0; i < semno; i++)
	semctl(semid_array[i], 0, IPC_RMID);

    shmctl(__shmid, IPC_RMID, NULL);
    msgctl(__msgid, IPC_RMID, NULL);
}
/*
static void __cleanup_zombie_descriptors(void)
{
}
*/
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
    struct gdev_sched_entity *se;

    printf("[%d]\t\tusers\tused/com_bw\tused/mem_bw\tcom_time\tmem_time\tparent\n",all_count++);
    printf("------------------------------------------------------------------------------\n");

    for(i = 0;i< gdev_count+gdev_vcount; i++){
	p= &(__gdevs[i]);
	if(i==((gdev_count-1)*gdev_vcount+(gdev_count-1))){
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
	se = &__se[i];
	if(se->list_entry_com.next!=0){
	    printf("se#%d!",i);
	}

    }
    printf("\n");
}


int main(void){
    int i;
    int attach_offset=0;


    /*  register the KILL signal.  */
    struct sigaction sa_kill;
    memset(&sa_kill, 0, sizeof(sa_kill));
    sigemptyset(&sa_kill.sa_mask);
    sa_kill.sa_handler = __kill_handler;
    sa_kill.sa_flags = 0;
    sigaction(SIGINT, &sa_kill, NULL);

    /* get the number of physical devices */
    gdev_count = 1; //fix this 


    /* get the number of virtual devices. */
    gdev_vcount = 0;
    for (i = 0; i < gdev_count; i++)
	gdev_vcount = 2; //fix this

    /* allocate Gdev all objects(shared memory).  */
    __shmid = __alloc_shm(GDEV_DEVICE_MAX_COUNT);

    /* attach gdev device objects */
    __attach_shm(__shmid, (void**)&__gdevs, 0, attach_offset);
    attach_offset += sizeof(*__gdevs) * GDEV_DEVICE_MAX_COUNT;
    for (i = 0; i < gdev_count + gdev_vcount; i++)
	__gdev_init_device(&(__gdevs[i]), i);

    GDEV_PRINT("Found %d physical device(s).\n", gdev_count);
    GDEV_PRINT("Configured %d virtual device(s).\n", gdev_vcount);

    __attach_shm(__shmid, (void**)&__vas, 0, attach_offset);
    attach_offset += sizeof(struct gdev_vas)* GDEV_CONTEXT_MAX_COUNT;
    for (i = 0; i< gdev_count + gdev_vcount; i++)
	__gdev_init_vas(&(__vas[i]));

    GDEV_PRINT("Prepare %d VAS entities(s).\n", GDEV_CONTEXT_MAX_COUNT);

    __attach_shm(__shmid, (void **)&__se, GDEV_CONTEXT_MAX_COUNT, attach_offset);
    attach_offset += sizeof(struct gdev_sched_entity)* GDEV_CONTEXT_MAX_COUNT;
    for (i = 0; i < GDEV_CONTEXT_MAX_COUNT; i++)
	__gdev_init_sched_entity(&(__se[i]), i);
    GDEV_PRINT("Prepared %d scheduling entities.\n",GDEV_CONTEXT_MAX_COUNT);


    __attach_shm(__shmid, (void **)&__mem, 256, attach_offset);
    attach_offset += sizeof(struct gdev_mem) * 256;

    GDEV_PRINT("Prepared %d memory entities.\n",256);

    //memset(__gdevs, 0, attach_offset);



    __init_gdev_monitor();

    /* clean up zombie descriptors periodically  */
    while(1){
	//      __cleanup_zombie_descriptors();
	__display();
	sleep(1);
    }

    /* shouldn't reach here */
    __exit_gdev_monitor();
    return 0;
}


