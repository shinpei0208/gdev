TARGET = kernel_test
$(TARGET)-y := main.o memcpy.o
GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
GDEVETC = $(GDEVDIR)/etc
DRMINC = /lib/modules/$(shell uname -r)/source/include/drm
CPPFLAGS += -Wno-strict-prototypes
KERNELDIR = /lib/modules/$(shell uname -r)/build

obj-m := $(TARGET).o

EXTRA_CFLAGS = -I $(GDEVINC)
KBUILD_EXTRA_SYMBOLS = $(GDEVETC)/Module.symvers.kcuda

all:
	+make -C $(KERNELDIR) M=$(PWD) modules

$(TARGET).o: $(OBJS)
	$(LD) $(LD_RFLAG) -r -o $@ $(OBJS)

clean:
	+make -C $(KERNELDIR) M=$(PWD) clean
	rm -f *~
