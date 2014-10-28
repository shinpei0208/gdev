# Using Gdev's user-space runtime library with PSCNV driver

Follow the instruction below to use Gdev's user-space runtime library
with PSCNV. You may be required to install additional software
packages depending on your environment. `$(TOPDIR)` represents your top
working directory.

## 1. Download

```sh
cd $(TOPDIR)
git clone git://github.com/shinpei0208/gdev.git # OR git clone git://github.com/CPFL/gdev.git
```

## 2. PSCNV Driver

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

## 3. Gdev Library

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
