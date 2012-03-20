#include <asm/uaccess.h>
#include <linux/kthread.h>
#include <linux/module.h>

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("CUDA Test");
MODULE_AUTHOR("Shinpei Kato");

int cuda_test_memcpy(uint32_t);

struct task_struct *test_thread = NULL;
static unsigned int size = 0x10000;
module_param(size, int, 0444);
MODULE_PARM_DESC(size, "data size (in hex)");

static void test_thread_func(void *__data)
{
	if (cuda_test_memcpy(size) < 0)
		printk("Test failed\n");
	else
		printk("Test passed\n");
}

static int __init cuda_test_init(void)
{
	test_thread = kthread_run((void*)test_thread_func, NULL, "cuda_test");
	return 0;
}

static void __exit cuda_test_exit(void)
{
}

module_init(cuda_test_init);
module_exit(cuda_test_exit);
