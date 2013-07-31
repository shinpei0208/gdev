include $(PWD)/API.mk

TARGET = kcuda
$(TARGET)-y := kcuda_drv.o init.o device.o version.o context.o module.o memory.o execution.o stream.o extension/ipc.o extension/memmap.o event.o gdev_cuda.o dummy.o
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
	@if [ -z "$(PWD)" ]; then \
		echo ""; \
		echo "should run 'make' without using 'sudo'."; \
		echo ""; \
		exit 1; \
	fi
	+make -C $(KERNELDIR) M=$(PWD) modules

clean:
	@if [ -z "$(PWD)" ]; then \
		echo ""; \
		echo "should run 'make clean' without using 'sudo'."; \
		echo ""; \
		exit 1; \
	fi
	+make -C $(KERNELDIR) M=$(PWD) clean
	rm -f *~

install: all
	@if [ -z "$(PWD)" ]; then \
		echo ""; \
		echo "should run 'make install' without using 'sudo'."; \
		echo ""; \
		exit 1; \
	fi
	-sudo rmmod $(TARGET)
	+sudo make -C $(KERNELDIR) M=$(PWD) modules_install
	sudo install -o root -m 0755 -d $(GDEVDIR)
	sudo install -o root -m 0755 -d $(GDEVETC)
	sudo install -o root -m 0644 Module.symvers \
	             $(GDEVETC)/Module.symvers.$(TARGET)
	sudo modprobe $(TARGET)

uninstall:
	@if [ -z "$(PWD)" ]; then \
		echo ""; \
		echo "should run 'make uninstall' without using 'sudo'."; \
		echo ""; \
		exit 1; \
	fi
	-sudo rmmod $(TARGET)
	sudo rm -f $(GDEVETC)/Module.symvers.$(TARGET)
	sudo rm -f /lib/modules/$(shell uname -r)/extra/$(TARGET).ko
	sudo depmod
