obj-m += gaes_ctr.o

all:
	cp ../../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp gaes_ctr.ko $(BUILD_DIR)/ ) 

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
