#include <asm/uaccess.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/vmalloc.h>

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("Gdev Kernel Test");
MODULE_AUTHOR("Shinpei Kato");

#define DATA_SIZE 0x20000000 /* 512MB */
#define CHUNK_SIZE 0x400000 /* 4MB */

int gdev_test_memcpy(uint32_t *in, uint32_t *out, uint32_t size, 
					 uint32_t chunk_size, int pipeline_count);

struct task_struct *test_thread = NULL;

static void test_thread_func(void *__data)
{
	uint32_t *in, *out;
	uint32_t size = DATA_SIZE;
	uint32_t ch_size = CHUNK_SIZE;
	int pl = 2;
	int i;

	in = (uint32_t *) vmalloc(size);
	out = (uint32_t *) vmalloc(size);

	for (i = 0; i < size / 4; i++) {
		in[i] = i+1;
		out[i] = 0;
	}

	gdev_test_memcpy(in, out, size, ch_size, pl);

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
