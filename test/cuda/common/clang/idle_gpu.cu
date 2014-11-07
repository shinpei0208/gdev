#include <stdint.h>
#include "clang/cuda.h"

extern "C"
__global__
void idle(unsigned int *p, unsigned int n)
{
    int x = __builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x()
          + __builtin_ptx_read_tid_x();
    int y = __builtin_ptx_read_ctaid_y() * __builtin_ptx_read_ntid_y()
          + __builtin_ptx_read_tid_y();
	unsigned int i = 0, j = 0, k = 0;
	__shared__ int s;

	s = *p;
	if (x == 0 && y == 0) {
		for (i = 0; i < n; i++) {
			if (x + y > n) {
				s = s + x;
				if (s > x + y)
					s = x;
			}
		}
	}
	*p = s;
}
