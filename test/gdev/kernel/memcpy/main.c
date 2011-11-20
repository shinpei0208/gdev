#include <asm/uaccess.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/vmalloc.h>

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("Gdev Kernel Test");
MODULE_AUTHOR("Shinpei Kato");

int gdev_test_memcpy(uint32_t *, uint32_t *, uint32_t);

struct task_struct *test_thread = NULL;

static void test_thread_func(void *__data)
{
	int i;
	uint32_t size = 0x400000; /* 4MB */
	uint32_t *in, *out;

start:
	in = (uint32_t *) vmalloc(size);
	out = (uint32_t *) vmalloc(size);

	for (i = 0; i < size / 4; i++) {
		in[i] = i;
		out[i] = 0;
	}

	gdev_test_memcpy(in, out, size);

	for (i = 0; i < size / 4; i++) {
		if (in[i] != out[i]) {
			printk("in[%d] = %u, out[%d] = %u\n", i, in[i], i, out[i]);
			printk("Test failed.\n");
			goto end;
		}
	}
	printk("Test passed.\n");
end:

	vfree(in);
	vfree(out);

	if (size != 0x10000000) {
		size *= 2;
		goto start;
	}
}

static int __init gdev_test_init(void)
{
	test_thread = kthread_create((void*)test_thread_func, NULL, "gdevtest");
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
