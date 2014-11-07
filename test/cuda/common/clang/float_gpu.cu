#include <stdint.h>
#include "clang/cuda.h"
extern "C"
__global__
void add(float a, float b, float *c)
{
	*c = a + b;
}
