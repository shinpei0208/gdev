# Gdev: Open-Source GPGPU Runtime and Driver Software

Follow the instruction below to use Gdev's user-space runtime library
with PSCNV. You may be required to install additional software
packages depending on your environment. `$(TOPDIR)` represents your top
working directory.

## Download

```sh
cd $(TOPDIR)
git clone git://github.com/CS005/gdev.git
```

## PSCNV Driver

Note that PSCNV works only for a limited version of the Linux kernel.
We have confirmed that the Linux kernel of version `2.6.{33-38}` is
available with PSCNV.

```sh
cd $(TOPDIR)/gdev/mod/linux/pscnv
./configure
make
sudo make modules_install
sudo shutdown -r now # will reboot your machine
```

## Gdev Library

Gdev's user-space library provides Gdev API. This API can be used
by either user programs directly or another high-level API library.
For instance, third party's CUDA library can use Gdev API.

```sh
cd $(TOPDIR)/gdev/lib
mkdir build
cd build
../configure --target=user
make
sudo make install
export LD_LIBRARY_PATH="/usr/local/gdev/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/gdev/bin:$PATH"
```


## Lincese
```
Copyright (C) Shinpei Kato

Nagoya University
Parallel and Distributed Systems Lab (PDSL)
http://pdsl.jp

University of California, Santa Cruz
Systems Research Lab (SRL)
http://systems.soe.ucsc.edu

All Rights Reserved.
```
