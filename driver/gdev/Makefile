TARGET := gdev
$(TARGET)-y := gdev_drv.o gdev_drv_nvidia.o gdev_fops.o gdev_ioctl.o gdev_proc.o
$(TARGET)-y += gdev_api.o gdev_device.o gdev_sched.o
$(TARGET)-y += gdev_nvidia.o gdev_nvidia_compute.o gdev_nvidia_mem.o gdev_nvidia_shm.o gdev_nvidia_nvc0.o

obj-m := $(TARGET).o

include $(PWD)/Driver.mk # export $(DRIVER_NAME)

EXTRA_CFLAGS := -Iinclude/drm -I/usr/local/gdev/$(DRIVER_NAME)
SYSSRC = /lib/modules/$(shell uname -r)/build
KBUILD_EXTRA_SYMBOLS = /usr/local/gdev/$(DRIVER_NAME)/Module.symvers

all:
	+make -C $(SYSSRC) M=$(PWD) modules

clean:
	+make -C $(SYSSRC) M=$(PWD) clean
	rm -f *~

install:
	@sh ./install.sh
