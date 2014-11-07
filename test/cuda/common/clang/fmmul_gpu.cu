#include <stdint.h>
#include "clang/cuda.h"
__global__
void mul(float *a, float *b, float *c, int n)
{
    int i = __builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x()
          + __builtin_ptx_read_tid_x();
    int j = __builtin_ptx_read_ctaid_y() * __builtin_ptx_read_ntid_y()
          + __builtin_ptx_read_tid_y();
    if (i < n && j < n) {
        int idx = i * n + j;
        c[idx] = a[idx] * b[idx];
    }
}
