#
# Makefile for the Linux 2.6 dm-crypt
#

obj-m += dm-crypt.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp dm-crypt.ko $(BUILD_DIR)/ ) 

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
