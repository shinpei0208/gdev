kapitest-y := fail.o \
	gamma_set_5.o gamma_set_6.o \
	drm_ioctl_def.o drm_ioctl_def_drv.o \
	drm_connector_detect_1.o drm_connector_detect_2.o \
	map_ofs.o \
	io_mapping_2.o io_mapping_3.o \
	i2c_id.o \
	pci_driver.o \
	switcheroo_reprobe.o \
	getparam_bus_type.o \
	drm_gem_object_handle_count.o 

obj-m := kapitest.o

EXTRA_CFLAGS = -Iinclude/drm

SYSSRC = /lib/modules/$(shell uname -r)/build

all:
	+make -C $(SYSSRC) M=$(PWD) modules

clean:
	+make -C $(SYSSRC) M=$(PWD) clean
