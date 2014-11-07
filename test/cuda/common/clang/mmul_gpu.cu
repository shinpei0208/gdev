#include "clang/cuda.h"

extern "C" __global__ void multiply(unsigned int *a, unsigned int *b, unsigned int *c,
 int n)
{
	unsigned int i;
	unsigned int product = 0;

	int row = __builtin_ptx_read_ctaid_y() * __builtin_ptx_read_ntid_y()
	        + __builtin_ptx_read_tid_y();
	int col = __builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x()
	        + __builtin_ptx_read_tid_x();
	
	if(row < n && col < n){
	    for (i = 0; i < n; i++)
		product += a[row * n + i] * b[i * n + col];

	    c[row*n + col] = product;
	}
}

