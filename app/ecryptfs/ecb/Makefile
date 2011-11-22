TARGET = ecb_gpu
GDEVDIR = /usr/local/gdev
GDEVINC = $(GDEVDIR)/include
GDEVETC = $(GDEVDIR)/etc
DRMINC = /lib/modules/$(shell uname -r)/source/include/drm
CPPFLAGS += -Wno-strict-prototypes
GPUSRC = aes_gpu
NVCC = nvcc -arch sm_20 -cubin
KERNELDIR = /lib/modules/$(shell uname -r)/build

obj-m := $(TARGET).o

EXTRA_CFLAGS = -I $(GDEVINC)
KBUILD_EXTRA_SYMBOLS = $(GDEVETC)/Module.symvers.kcuda

all:
	@echo '#define DIRPATH "$(PWD)"' > dirpath.h
	$(NVCC) -o $(GPUSRC).cubin $(GPUSRC).cu
	+make -C $(KERNELDIR) M=$(PWD) modules

clean:
	+make -C $(KERNELDIR) M=$(PWD) clean
	rm -f dirpath.h *.cubin *~
