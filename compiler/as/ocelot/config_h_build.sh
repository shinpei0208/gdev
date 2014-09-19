#!/bin/sh
if [ $# -ne 2 ]; then
	echo "Usage: $0 configure.h.in configure.h" 1>&2
	exit 1
fi

sed -e 's/\%(HAVE_LLVM)d/0/g' \
    -e 's/\%(EXCLUDE_CUDA_RUNTIME)d/0/g' \
    -e 's/\%(HAVE_GLEW)d/0/g' \
    -e 's/\%(VERSION)s/2.1.unknown/g' < $1 > $2
