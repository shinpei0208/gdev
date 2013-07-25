TARGET := gdev
$(TARGET)-y := gdev_drv.o gdev_drv_nvidia.o gdev_fops.o gdev_ioctl.o gdev_proc.o
$(TARGET)-y += gdev_api.o gdev_device.o gdev_sched.o
$(TARGET)-y += gdev_nvidia.o gdev_nvidia_fifo.o gdev_nvidia_compute.o gdev_nvidia_mem.o gdev_nvidia_shm.o gdev_nvidia_nvc0.o gdev_nvidia_nve4.o

obj-m := $(TARGET).o

include $(PWD)/Driver.mk # export $(DRIVER_NAME)

SYSSRC = /lib/modules/$(shell uname -r)/build

GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
GDEVETC = $(GDEVDIR)/etc

ifeq ($(wildcard ($(SYSSRC)/drivers/gpu/drm/nouveau/gdev_interface.h)),)
	GDEV_INTERFACE = $(SYSSRC)/drivers/gpu/drm/nouveau
else
	GDEV_INTERFACE = $(GDEVDIR)/$(DRIVER_NAME)
	KBUILD_EXTRA_SYMBOLS = $(GDEV_INTERFACE)/Module.symvers
endif

EXTRA_CFLAGS = -Iinclude/drm -I$(GDEV_INTERFACE) -DGDEV_SCHED_DISABLED

all:
	+make -C $(SYSSRC) M=$(PWD) modules

clean:
	+make -C $(SYSSRC) M=$(PWD) clean
	rm -f *~

install: all
	-sudo rmmod $(TARGET)
	+sudo make -C $(SYSSRC) M=$(PWD) modules_install
	sudo install -o root -m 0755 -d $(GDEVDIR)
	sudo install -o root -m 0755 -d $(GDEVINC)
	sudo install -o root -m 0755 -d $(GDEVETC)
	sudo install -o root -m 0644 Module.symvers \
	             $(GDEVETC)/Module.symvers.$(TARGET)
	sudo install -o root -m 0644 gdev_api.h gdev_autogen.h \
	             gdev_nvidia_def.h gdev_list.h gdev_time.h $(GDEVINC)
	sudo install -o root -m 0644 $(TARGET).rules \
	             /etc/udev/rules.d/80-$(TARGET).rules
	-sudo modprobe $(TARGET)

uninstall:
	-sudo rmmod $(TARGET)
	sudo rm -f $(GDEVETC)/Module.symvers.$(TARGET)
	sudo rm -f $(GDEVINC)/gdev_api.h $(GDEVINC)/gdev_autogen.h \
	        $(GDEVINC)/gdev_nvidia_def.h $(GDEVINC)/gdev_list.h \
	        $(GDEVINC)/gdev_time.h
	sudo rm -f /lib/modules/$(shell uname -r)/extra/$(TARGET).ko
	sudo depmod
