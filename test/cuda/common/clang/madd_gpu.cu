#include <stdint.h>
#include "clang/cuda.h"
__global__
void add(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t n)
{
    int i = __builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x()
          + __builtin_ptx_read_tid_x();
    int j = __builtin_ptx_read_ctaid_y() * __builtin_ptx_read_ntid_y()
          + __builtin_ptx_read_tid_y();
    if (i < n && j < n) {
        int idx = i * n + j;
        c[idx] = a[idx] + b[idx];
    }
}
