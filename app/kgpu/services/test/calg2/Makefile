obj-m += calg2.o

all:
	cp ../../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp calg2.ko $(BUILD_DIR)/ )

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
