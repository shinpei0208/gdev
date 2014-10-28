# Building/Using Gdev with CMake

Follow the instruction below to use Gdev with OS runtime support. You
may be required to install additional software packages depending on
your environment. `$(TOPDIR)` represents your top working directory.

## 1. Download

Check out the Gdev and envytools source code via `git`.

```sh
cd $(TOPDIR)
git clone git://github.com/shinpei0208/gdev.git  # OR git://github.com/CPFL/gdev.git
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

## 3. Building Gdev

Gdev provides the build instruction with CMake.
Leveraging CMake, you can build/install Gdev and CUDA runtime.
The following options are supported.

| option | description | values |
| :-- | :-- | :-- |
| driver |select GPU driver. must set driver name in the user-space mode| `nouveau`,`pscnv`,`nvrm`,`barra` |
| user | user mode (default off) | ON/OFF |
| runtime | enable CUDA runtime API (default on)| ON/OFF |
| usched | enable user mode scheduler (default off)| ON/OFF |
| use\_as | use assembler (default off)| ON/OFF |

For example, you can pass the above options to CMake like below.

```sh
cmake -H. -Brelease \
    -Ddriver=nouveau \
    -Duser=ON \
    -Druntime=ON \
    -Dusched=OFF \
    -Duse_as=OFF \
    -DCMAKE_BUILD_TYPE=Release
make -C release
make -C release install
```

This will mainly produce

1. user-space mode Gdev `libgdev.so` and
2. a limited set of CUDA Driver/Runtime API `libucuda.so`

`make` builds them with the specified options and they are installed under the directory `/usr/local/gdev`.

__CAUTION__:
Especially, the libraries are installed under the directory `/usr/local/gdev/lib64`.
So please remember to add `/usr/local/gdev/lib64` to `$LD_LIBRARY_PATH`.

## 4. CUDA Driver API test (user-space programs)

You can build / test examples using CUDA Driver API. They are located under the directory `$(TOPDIR)/test/cuda/user/`.
Follow the instruction to build and use the examples.

```sh
cd $(TOPDIR)/test/cuda/user/madd
make
./user_test 256 # a[256] + b[256] = c[256]
```
