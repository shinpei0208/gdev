# Using Gdev's user-space runtime library with Nouveau driver

Follow the instruction below to use Gdev's user-space runtime library
with Nouveau. You may be required to install additional software
packages depending on your environment. `$(TOPDIR)` represents your top
working directory.

## 1. Download

```sh
cd $(TOPDIR)
git clone git://github.com/shinpei0208/gdev.git # OR git clone git://github.com/CPFL/gdev.git
```

## 2. Linux Kernel and Nouveau Device Driver

To use Gdev's user-space runtime library, you need to install a
native GPU device driver.

```sh
cd $(TOPDIR)
git clone --depth 1 git://anongit.freedesktop.org/nouveau/linux-2.6
cd linux-2.6
git remote add nouveau git://anongit.freedesktop.org/nouveau/linux-2.6
git remote update
git checkout -b nouveau-master nouveau/master
make oldconfig
make
sudo make modules_install install
sudo shutdown -r now # will reboot your machine
modprobe -r nouveau; modprobe nouveau modeset=1 noaccel=0
```

## 3. Gdev Library

Gdev's user-space library provides Gdev API. This API can be used
by either user programs directly or another high-level API library.
For instance, third party's CUDA libraries may use Gdev API.

The Gdev user-space runtime library requires Nouveau's LIBDRM.
You would need to install some distro packages to build.
e.g, `autoconf`, `automake`, `libtool`, `libxcb-devel`

```sh
cd $(TOPDIR)
git clone git://anongit.freedesktop.org/git/mesa/drm/
cd drm
./autogen.sh
./configure --enable-nouveau-experimental-api --prefix=/usr/
make
sudo make install
```

Now you build the Gdev user-space runtime library as follows:

```sh
cd $(TOPDIR)/gdev/lib
mkdir build
cd build
../configure --target=user
sudo make install
export LD_LIBRARY_PATH="/usr/local/gdev/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/gdev/bin:$PATH"
```
