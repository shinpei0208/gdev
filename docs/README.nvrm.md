# Gdev: Open-Source GPGPU Runtime and Driver Software

Follow the instruction below to use Gdev with OS runtime support. You
may be required to install additional software packages depending on
your environment. `$(TOPDIR)` represents your top working directory.

## Download

```sh
cd $(TOPDIR)
git clone git://github.com/CS005/gdev.git
wget http://www.ertl.jp/~shinpei/download/nvidia/NVIDIA-Linux-x86_64-313.18.run
# the latest versions of the NVIDIA binary driver should also work.
```

## Device Driver

Gdev disgregates from the device driver. You need to install a native
GPU device driver to use Gdev.

```sh
cd $(TOPDIR)
sudo init 3
sudo sh NVIDIA-Linux-x86_64-313.18.run
sudo init 5
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
