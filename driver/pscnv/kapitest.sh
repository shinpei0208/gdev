#!/bin/sh
TESTS="gamma_set_5 gamma_set_6 drm_ioctl_def drm_ioctl_def_drv drm_connector_detect_1 drm_connector_detect_2 map_ofs io_mapping_2 io_mapping_3 i2c_id switcheroo_reprobe getparam_bus_type drm_gem_object_handle_count drm_get_dev drm_init fb_info_apertures drm_driver_fops noop_llseek drm_mode_fb_cmd2 drm_fb_pitch"

make -k -C $1 M=$PWD/kapitest clean 2> /dev/null 1> /dev/null 
make -k -C $1 M=$PWD/kapitest modules 2> /dev/null 1> /dev/null

for i in $TESTS
do
	if [ -f kapitest/$i.o ]
	then
		echo \#define PSCNV_KAPI_`echo $i | tr a-z A-Z`
	fi
done
