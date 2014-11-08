# Gdev: Open-Source GPGPU Runtime and Driver Software

[![Build Status](https://travis-ci.org/shinpei0208/gdev.svg?branch=master)](https://travis-ci.org/shinpei0208/gdev)

Gdev is a rich set of open-source software for NVIDIA GPGPU technology,
containing device drivers, CUDA runtimes, CUDA/PTX compilers, and some
utility tools. Currently it only supports NVIDIA GPUs and Linux but is,
by design, portable to other GPUs and platforms as well.
The supported API implementaions include:

- __Gdev API__: A low-level API to manage details of GPUs.
- __CUDA Driver API__: A low-level API adovocated by NVIDIA.
- __CUDA Runtime API__: A high-level API adovocated by NVIDIA.

The implementation of CUDA Driver API and CUDA Runtime API is built
on top of Gdev API. For CUDA Runtime API we make use of GPU Ocelot as
a front-end implementation. You can add your favorite high-level API
to Gdev other than CUDA Driver/Runtime APIs on top of Gdev API.

Gdev provides runtime support in both the device driver and the user-
space library. Device-driver runtime support is a unique feature of
Gdev while most existing GPGPU programming frameworks take user-space
approaches. With device-driver runtime support, Gdev allows the OS to
manage GPUs as first-class citizens and execute CUDA programs itself.
Gdev's user-space runtime support is also unique in a sense that it
is available for multiple open-source and proprietary device drivers.
The supported device drivers include:

- __Nouveau__: An open-source driver developed by the Linux community.
- __PSCNV__: An open-source driver developed by PathScale.
- __NVRM__: A proprietary binary driver provided by NVIDIA.

To summarize, Gdev offers the following advantages:

- You have open-source access to GPGPU runtime and driver software.
- You can execute CUDA in the OS using loabable kernel modules.
- You can investigate GPU resource management in research.
- You can enhance OS and user-space runtime support capabilities.
- You can compare device drivers performance under the same runtime.

## How to download, install, and use Gdev

The recommended way to build/install Gdev is building/installing it with CMake.
- See [docs/README.cmake.md](/docs/README.cmake.md)

Otherwise, you can choose one of the following for what driver to be used (__obsolete__):

1. Do you want to use runtime support in the OS?
    - See [docs/README.gdev.md](/docs/README.gdev.md)
2. Do you want to use user-space runtime with Nouveau?
    - See [docs/README.nouveau.md](/docs/README.nouveau.md)
3. Do you want to use user-space runtime with PSCNV?
    - See [docs/README.pscnv.md](/docs/README.pscnv.md)
4. Do you want to use user-space runtime with NVRM (NVIDIA Driver)?
    - See [docs/README.nvrm.md](/docs/README.nvrm.md)


Once the driver is successfully installed, you can install high-level API:

1. Do you want to use CUDA?
    - See [docs/README.cuda.md](/docs/README.cuda.md)

## The publication of the Gdev project

1. S. Kato, M. McThrow, C. Maltzahn, and S. Brandt. "Gdev: First-Class GPU Resource Management in the Operating System", In Proceedings of the 2012 USENIX Annual Technical Conference (USENIX ATC'12), 2012.

## Related research papers

1. Y. Abe, H. Sasaki, M. Peres, K. Inoue, K. Murakami, and S. Kato. "Power and Performance Analysis of GPU-Accelerated Systems", In Proceedings of the 5th UESNIX Workshop on Power-Aware Computing and Systems (HotPower'12) , 2012.
2. S. Kato. "Implementing Open-Source CUDA Runtime", In Proceedings of the 54th Programming Symposium, Jan, 2013.
3. S. Kato, J. Aumiller, and S. Brandt. "Zero-Copy I/O Processing for Low-Latency GPU Computing", In Proceedings of the 4th ACM/IEEE International Conference on Cyber-Physical Systems (ICCPS'13), 2013.
4. Y. Fujii, T. Azumi, N. Nishio, and S. Kato. "Exploring Microcontrollers in GPUs", In Proceedings of the 4th Asia-Pacific Workshop on Systems, (APSys'13), 2013.

## Reclocking the GPU

NVIDIA's graphics cards are set very low clocks by default. To get
performance, you need to reclock your card at the maximum level.
How? Be root first, and then echo `3` to the following file:

```sh
echo 3 > /sys/class/drm/card0/device/performance_level
```

You can downclock your card by echoing `0` to the same file, i.e.,

```sh
echo 0 > /sys/class/drm/card0/device/performance_level
```

There are middleground levels `1` and `2`, too. Note that Reclocking
is not completely supported by the open-source solution yet.
There are still some performance levels missing, and hence you may
not get as high performance as the blob. If you really need the same
level of performance as the blob, you can run some long-running
CUDA program with the blob, and do `kexec -f your kernel` before the
program is finished. Then the clock remains at the maximum level.


## Benchmarks and Applications

Today many CUDA programs are written using CUDA Runtime API. If you
want to test CUDA Driver API, try the following benchmarks and apps.

- `git@github.com:shinpei0208/gdev-app.git`
- `git@github.com:shinpei0208/gdev-bench.git`


## Contributors

- Yuki ABE, Kyushu University
- Jason AUMILLER, University of California at Santa Cruz
- Takuya AZUMI, Ritsumeikan University
- Masato EDAHIRO, Nagoya University
- Yusuke FUJII, Ritsumeikan University
- Tsuyoshi HAMADA, Nagasaki University
- Masaki IWATA, AXE Inc.
- Shinpei KATO, Nagoya University (Maintainer)
- Marcin KOSCIELNICKI, University of Warsaw
- Michael MCTHROW, University of California at Santa Cruz
- Martin PERES, University of Bordeaux
- Hiroshi SASAKI, Kyushu University
- Yusuke SUZUKI, Keio University
- Hisashi USUDA, AXE Inc.
- Kaibo WANG, Ohio State University
- Hiroshi YAMADA, Tokyo University of Agriculture and Technology


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
