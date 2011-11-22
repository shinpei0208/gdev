TARGET = kernel_test
$(TARGET)-y = main.o memcpy.o
GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
GDEVETC = $(GDEVDIR)/etc
DRMINC = /lib/modules/$(shell uname -r)/source/include/drm
CPPFLAGS += -Wno-strict-prototypes
KERNELDIR = /lib/modules/$(shell uname -r)/build

obj-m := $(TARGET).o

EXTRA_CFLAGS := -I$(GDEVINC) -I$(DRMINC)
KBUILD_EXTRA_SYMBOLS = $(GDEVETC)/Module.symvers.gdev

all:
	+make -C $(KERNELDIR) M=$(PWD) modules

clean:
	+make -C $(KERNELDIR) M=$(PWD) clean
	rm -f *~

