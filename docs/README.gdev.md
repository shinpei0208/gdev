# Using Gdev

Follow the instruction below to use Gdev with OS runtime support. You
may be required to install additional software packages depending on
your environment. `$(TOPDIR)` represents your top working directory.

## 1. Download

```sh
cd $(TOPDIR)
git clone git://github.com/shinpei0208/gdev.git # OR git://github.com/CPFL/gdev.git
git clone git://github.com/envytools/envytools.git
```

## 2. envytools

envytools is a rich set of open-source tools to compile or decompile
NVIDIA GPU program code, firmware code, macro code, and so on. It is
also used to generate C header files with GPU command definitions.
In addition, envytools document the NVIDIA GPU architecture details,
while are not disclosed to the public so far. If you are interested
in GPU system software development, this is what you should read!
Please follow the instruction below to install envytools.

```sh
cd $(TOPDIR)/envytools
mkdir build
cd build
cmake .. # may require some packages on your distro
make
sudo make install # will install tools to /usr/local/{bin,lib}
```

## 3. Linux Kernel and Nouveau Device Driver

Gdev disgregates from the device driver. You need to install a native
GPU device driver to use Gdev.

__CAUTION__: If you would like to use the user mode gdev, this kernel is not necessary.

```sh
cd $(TOPDIR)
git clone --depth 1 git://anongit.freedesktop.org/nouveau/linux-2.6
cd linux-2.6
git remote add nouveau git://anongit.freedesktop.org/nouveau/linux-2.6
git remote update
git checkout -b nouveau-master nouveau/master
patch -p1 < $(TOPDIR)/gdev/mod/linux/patches/gdev-nouveau-X.X.patch
make oldconfig
make
sudo make modules_install install
sudo shutdown -r now # will reboot your machine
modprobe -r nouveau; modprobe nouveau modeset=1 noaccel=0
```

## 4. Gdev Kernel Module

This is a main module of Gdev providing OS runtime support.

__CAUTION__: If you would like to use the user mode gdev, this kernel module is not necessary.

```sh
cd $(TOPDIR)/gdev/mod
mkdir build
cd build
../configure
make
make install
```

## 5. Gdev Library

Since this version of Gdev provides runtime support in the OS, this
library is just a set of wrapper functions that call the Gdev module
functions via ioctl.

```sh
cd $(TOPDIR)/gdev/lib
mkdir build
cd build
../configure
make
sudo make install
export LD_LIBRARY_PATH="/usr/local/gdev/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/gdev/bin:$PATH"
```
