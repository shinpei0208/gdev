#!/bin/sh

x_exist=$(ps aux | grep /bin/X | wc -l)

if [ $x_exist -gt 1 ]; then
	echo "Error: X is running"
	echo "Please exit X before installing Gdev"
	exit
fi

nvidia_exist=$(lsmod | grep nvidia | wc -l)
nouveau_exist=$(lsmod | grep nouveau | wc -l)
ttm_exist=$(lsmod | grep ttm | wc -l)

if [ $nvidia_exist -gt 0 ]; then
	rmmod nvidia
fi
if [ $nouveau_exist -gt 0 ]; then
	rmmod nouveau
fi
if [ $ttm_exist -gt 0 ]; then
	rmmod ttm
fi

depmod $(pwd)/pscnv.ko
modprobe pscnv

sh ./gdev.sh
