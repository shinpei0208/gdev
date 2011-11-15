#!/bin/sh

rmmod nvidia
rmmod nouveau
rmmod ttm
insmod ./gdev.ko

# get a proper group
group="staff"
grep -q '^staff:' /etc/group || group="wheel"

# remove stale nodes
rm -f /dev/gdev*

# make nodes
major=$(awk "\$2==\"gdev\" {print \$1}" /proc/devices)
devnum=$(ls /dev/dri/card* | awk "{print \$1}" | wc -l)
minor=0
while [ $minor -lt $devnum ]
do
        mknod /dev/gdev${minor} c $major $minor
        chgrp $group /dev/gdev${minor}
        #chmod 664 /dev/gdev${minor}
        chmod 666 /dev/gdev${minor} # just temporarily
        minor=$(expr $minor + 1)
done

gdevdir=/usr/local/gdev
gdevinc=$gdevdir/include
# install header files
if [ ! -d $gdevdir ]; then
	mkdir $gdevdir
fi
if [ ! -d $gdevinc ]; then
	mkdir $gdevinc
fi
cp -f Module.symvers $gdevdir
cp -f {gdev_api.h,gdev_drv.h,gdev_nvidia_def.h,gdev_list.h,gdev_time.h} $gdevinc

