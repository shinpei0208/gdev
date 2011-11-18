obj-m += callgpu.o

all:
	cp ../../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp callgpu.ko $(BUILD_DIR)/ )

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
