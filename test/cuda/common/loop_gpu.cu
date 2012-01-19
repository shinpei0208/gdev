#include <stdint.h>
#include <cuda.h>

__global__
void loop(uint32_t *data, uint32_t size, uint32_t n)
{
    int i;
//	for (i = 0; i < n/40; i++) {
	for (i = 0; i < n/5; i++) {
		if (i * 4 < size)
			data[i] = i + n;
    }
}
