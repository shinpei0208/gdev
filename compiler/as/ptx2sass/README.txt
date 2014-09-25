To build ptx2sass:

1. Type `cmake -Duse_as=ON .' at the top level directory of Gdev
   or type `cmake .' at the current directory to build the Makefile.

2. Type `make' to compile the libocelot and ptx2sass.

3. Type `sudo make install' to install the ptx2sass to
   /usr/local/gdev/bin.

To use ptx2sass:

  $ ptx2sass -i ptx_file -o sass_file

for example:

  $ nvcc -ptx -arch=sm_30 -m64 -o foo.ptx foo.cu
  $ ptx2sass -i foo.ptx -o foo.sass
  $ sass2cubin foo.sass -o foo.cubin
