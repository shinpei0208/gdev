# Using Gdev's CUDA

Follow the instruction below to use Gdev's CUDA. You may be required
to install additional software packages depending on your environment.
e.g., `bison`, `flex`, `boost`, `boost-devel`

`$(TOPDIR)` represents your top working directory where you installed
the Gdev repository.

## 1. Gdev's CUDA installation

Gdev currently supports a limited set of CUDA Driver/Runtime API.
It is defined as Micro CUDA (uCUDA) in Gdev.

```sh
cd $(TOPDIR)/gdev/cuda
mkdir build
cd build
#if you want to disable Runtime API
../configure --disable-runtime
#else if you want to use a PTX parser of GPU Ocelot
../configure --with-ptx
#else
../configure
#endif
make
sudo make install
```

Gdev also supports CUDA in the operating system. You are required to
install `kcuda` module to use this functionality.
- See [docs/README.gdev.md](/docs/README.gdev.md)

__Note__: that kcuda supports CUDA Driver API only.

```sh
cd $(TOPDIR)/gdev/cuda
mkdir kbuild
cd kbuild
../configure --target=kcuda
make
sudo make install
```

## 2. CUDA Driver API test (user-space programs)

```sh
cd $(TOPDIR)/test/cuda/user/madd
make
./user_test 256 # a[256] + b[256] = c[256]
```

## 3. CUDA Driver API test (kernel-space programs)

Note that you need `kcuda` to be installed a priori if you want to
run CUDA in the OS.

```sh
cd $(TOPDIR)/test/cuda/kernel/memcpy
make
sudo insmod ./kernel_test.ko size=10000 # copy 0x10000 size
```

NOTE: Please be careful when doing this test as it runs a program
in `module_init()`. If you run a very long program as it is, you may
crash your system. If you want to run a very long program, you must
provide a proper module implementation, e.g., using kernel threads.
