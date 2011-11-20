TARGET = kcuda
$(TARGET)-y := kcuda_drv.o init.o device.o version.o context.o module.o memory.o execution.o gdev_cuda.o
GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
GDEVETC = $(GDEVDIR)/etc
DRMINC = /lib/modules/$(shell uname -r)/source/include/drm
CPPFLAGS += -Wno-strict-prototypes
KERNELDIR = /lib/modules/$(shell uname -r)/build

obj-m := $(TARGET).o

EXTRA_CFLAGS = -I $(GDEVINC) -I $(DRMINC)
KBUILD_EXTRA_SYMBOLS = $(GDEVETC)/Module.symvers.gdev

all:
	+make -C $(KERNELDIR) M=$(PWD) modules

clean:
	+make -C $(KERNELDIR) M=$(PWD) clean
	rm -f *~

install:
	cp -f Module.symvers $(GDEVETC)/Module.symvers.kcuda
	insmod $(TARGET).ko

uninstall:
	rmmod $(TARGET)
	rm -f $(GDEVETC)/Module.symvers.kcuda
