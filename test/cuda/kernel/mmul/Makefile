TARGET = kernel_test
$(TARGET)-y := main.o mmul.o
GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
GDEVETC = $(GDEVDIR)/etc
DRMINC = /lib/modules/$(shell uname -r)/source/include/drm
CPPFLAGS += -Wno-strict-prototypes
KERNELDIR = /lib/modules/$(shell uname -r)/build
NVCC = nvcc -arch sm_20 -cubin

obj-m := $(TARGET).o

EXTRA_CFLAGS = -I $(GDEVINC)
KBUILD_EXTRA_SYMBOLS = $(GDEVETC)/Module.symvers.kcuda

all:
	@echo '#define DIRPATH "$(PWD)"' > dirpath.h
	$(NVCC) -o mmul_gpu.cubin mmul_gpu.cu
	+make -C $(KERNELDIR) M=$(PWD) modules

clean:
	+make -C $(KERNELDIR) M=$(PWD) clean
	rm -f *~ *.cubin dirpath.h
