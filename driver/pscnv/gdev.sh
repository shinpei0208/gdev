#!/bin/sh

gdevdir=/usr/local/gdev

if [ ! -d $gdevdir ]; then
        mkdir $gdevdir
fi

pscnvdir=$gdevdir/pscnv

if [ ! -d $pscnvdir ]; then
        mkdir $pscnvdir
fi

cp gdev_interface.h Module.symvers $pscnvdir
