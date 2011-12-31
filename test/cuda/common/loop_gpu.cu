#include <stdint.h>
#include <cuda.h>

__global__
void loop(uint32_t *data, uint32_t size, uint32_t n)
{
    int i;
    for (i = 0; i < n/400; i++) {
		if (i * 4 < size)
			data[i] = i + n;
    }
}
