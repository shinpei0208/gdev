#include <linux/proc_fs.h>
#include "gdev_drv.h"

#define GDEV_PROC_MAX_BUF 64

static struct proc_dir_entry *gproc_dir;
static struct semaphore gproc_sem;

static int gdev_proc_read(char *kbuf, char *page, int count, int *eof)
{
	down(&gproc_sem);
	//strncpy(page, kbuf, count);
	sprintf(page, "%s", kbuf);
	count = strlen(page);
	*eof = 1;
	up(&gproc_sem);
	return count;
}

static int gdev_proc_write(char *kbuf, const char *buf, int count)
{
	down(&gproc_sem);
	if (count > GDEV_PROC_MAX_BUF - 1)
		count = GDEV_PROC_MAX_BUF - 1;
	if (copy_from_user(kbuf, buf, count)) {
		GDEV_PRINT("Failed to write /proc entry\n");
		up(&gproc_sem);
		return -EFAULT;
	}
	up(&gproc_sem);
	return count;
}

static int device_count_read
(char *page, char **start, off_t off, int count, int *eof, void *data)
{
	char kbuf[64];
	sprintf(kbuf, "%d", gdrv.count);
	return gdev_proc_read(kbuf, page, count, eof);
}

static int device_count_write
(struct file *filp, const char __user *buf, unsigned long count, void *data)
{
	char kbuf[64];
	count = gdev_proc_write(kbuf, buf, count);
	sscanf(kbuf, "%d", &gdrv.count);
	return count;
}

int gdev_create_proc(void)
{
	struct proc_dir_entry *proc_device_count;

	gproc_dir = proc_mkdir("gdev", NULL);
	if (!gproc_dir) {
		GDEV_PRINT("Failed to create /proc entry\n");
		return 0;
	}

	proc_device_count = create_proc_entry("device_count", 0644, gproc_dir);
	if (!proc_device_count) {
		GDEV_PRINT("Failed to create /proc entry\n");
		return 0;
	}
	proc_device_count->read_proc = device_count_read;
	proc_device_count->write_proc = device_count_write;
	proc_device_count->data = (void*) NULL;

	sema_init(&gproc_sem, 1);

	return 0;
}

int gdev_delete_proc(void)
{
	remove_proc_entry("gdev/device_count", gproc_dir);
	remove_proc_entry("gdev", NULL);

	return 0;
}

