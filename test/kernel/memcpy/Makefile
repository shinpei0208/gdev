TARGET = kernel_test
OBJS = main.o memcpy.o
GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
DRMINC = /lib/modules/$(shell uname -r)/source/include/drm
CPPFLAGS += -Wno-strict-prototypes

# If KERNELRELEASE is define, we have been invoked from the
# kernel build system and can use its languages.
ifneq ($(KERNELRELEASE),)
	obj-m := $(TARGET).o
	$(TARGET)-objs := $(OBJS)
	EXTRA_CFLAGS := -I$(GDEVINC) -I$(DRMINC)

# Otherwise we were called directly from the command line;
# invoke the kernel build system
else
	KERNELDIR ?= /lib/modules/$(shell uname -r)/build
	EXTRA_CFLAGS := -I$(GDEVINC) -I$(DRMINC)

all:
	@cp -f $(GDEVDIR)/Module.symvers .
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

$(TARGET).o: $(OBJS)
	$(LD) $(LD_RFLAG) -r -o $@ $(OBJS)

clean:
	@rm -fr .tmp* .*.o.cmd *.mod.* *.ko *.o .$(TARGET)* modules.order Module.* *~

.PHONY:install
install:
	@./script/install

.PHONY:uninstall
uninstall:
	@./script/uninstall
endif
