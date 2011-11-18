#include <asm/uaccess.h>
#include <linux/kthread.h>
#include <linux/module.h>

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("Gdev Kernel Test");
MODULE_AUTHOR("Shinpei Kato");

int gdev_test_loadstore(void);

struct task_struct *test_thread = NULL;

static void test_thread_func(void *__data)
{
	int ret = gdev_test_loadstore();
	if (ret < 0)
		printk("Test failed\n");
	else
		printk("Test passed\n");
	while (!kthread_should_stop()) {
		schedule_timeout(1000);
	}
}	

static int __init gdev_test_init(void)
{
	test_thread = kthread_create((void*)test_thread_func, NULL, "gdevtest");
	/*sched_setscheduler(test_thread, SCHED_FIFO, &sp);*/
	wake_up_process(test_thread);
	return 0;
}

static void __exit gdev_test_exit(void)
{
	if (test_thread) {
		kthread_stop(test_thread);
	}
}

module_init(gdev_test_init);
module_exit(gdev_test_exit);
