#include <stdint.h>
#include <cuda.h>

extern "C"
__global__
void idle(unsigned int *p, unsigned int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
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
