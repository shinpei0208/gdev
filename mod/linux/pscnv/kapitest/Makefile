kapitest-y := fail.o \
	gamma_set_5.o gamma_set_6.o \
	drm_ioctl_def.o drm_ioctl_def_drv.o \
	drm_connector_detect_1.o drm_connector_detect_2.o \
	map_ofs.o \
	io_mapping_2.o io_mapping_3.o \
	i2c_id.o \
	switcheroo_reprobe.o \
	getparam_bus_type.o \
	drm_init.o \
	drm_gem_object_handle_count.o \
	drm_get_dev.o \
	fb_info_apertures.o \
	drm_driver_fops.o \
	noop_llseek.o \
	drm_mode_fb_cmd2.o \
	drm_fb_pitch.o

obj-m := kapitest.o

EXTRA_CFLAGS = -Iinclude/drm

SYSSRC = /lib/modules/$(shell uname -r)/build

all:
	+make -C $(SYSSRC) M=$(PWD) modules

clean:
	+make -C $(SYSSRC) M=$(PWD) clean
