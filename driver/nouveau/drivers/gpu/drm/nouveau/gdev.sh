#!/bin/sh

gdevdir=/usr/local/gdev

if [ ! -d $gdevdir ]; then
        mkdir $gdevdir
fi

nouveaudir=$gdevdir/nouveau

if [ ! -d $nouveaudir ]; then
        mkdir $nouveaudir
fi

cp gdev_interface.h ../Module.symvers $nouveaudir
