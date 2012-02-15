#!/bin/sh

nvidia_exist=$(lsmod | grep nvidia | wc -l)
nouveau_exist=$(lsmod | grep nouveau | wc -l)
ttm_exist=$(lsmod | grep ttm | wc -l)
drm_exist=$(lsmod | grep drm | wc -l)
drm_kms_helper_exist=$(lsmod | grep drm_kms_helper | wc -l)
i2c_algo_bit_exist=$(lsmod | grep i2c_algo_bit_exist | wc -l)

if [ $nvidia_exist -eq 1 ]; then
	rmmod nvidia
fi
if [ $nouveau_exist -eq 1 ]; then
	rmmod nouveau
fi
if [ $ttm_exist -eq 1 ]; then
	rmmod ttm
fi
if [ ! $drm_exist -eq 1 ]; then
	modprobe drm
fi
if [ ! $drm_kms_helper_exist -eq 1 ]; then
	modprobe drm_kms_helper
fi
if [ ! $i2c_algo_bit_exist -eq 1 ]; then
	modprobe i2c_algo_bit
fi

exit

insmod ./gdev.ko

# get a proper group
group="staff"
grep -q '^staff:' /etc/group || group="wheel"

# remove stale nodes
rm -f /dev/gdev*

# make nodes
major=$(awk "\$2==\"gdev\" {print \$1}" /proc/devices)
devnum=$(less /proc/gdev/virtual_device_count | awk "{print \$1}")
minor=0
while [ "$minor" -lt "$devnum" ]
do
        mknod /dev/gdev${minor} c $major $minor
        chgrp $group /dev/gdev${minor}
        chmod 666 /dev/gdev${minor}
        minor=$(expr $minor + 1)
done

gdevdir=/usr/local/gdev
gdevinc=$gdevdir/include
gdevetc=$gdevdir/etc
# install header files
if [ ! -d $gdevdir ]; then
	mkdir $gdevdir
fi
if [ ! -d $gdevinc ]; then
	mkdir $gdevinc
fi
if [ ! -d $gdevetc ]; then
	mkdir $gdevetc
fi
cp -f Module.symvers $gdevetc/Module.symvers.gdev
cp -f gdev_api.h gdev_nvidia_def.h gdev_list.h gdev_time.h $gdevinc

