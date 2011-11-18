#
# Makefile for the Linux 2.6 eCryptfs
#

obj-m += ecryptfs.o

ecryptfs-objs := dentry.o file.o inode.o main.o super.o mmap.o read_write.o crypto.o keystore.o messaging.o miscdev.o kthread.o debug.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp ecryptfs.ko $(BUILD_DIR)/ ) 

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
