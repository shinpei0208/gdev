/*
 * madd_gpu.cu -- Device code for matrix additon benchmark
 *
 * Michael McThrow
 */

#define get_element_index(i, j, cols) ((i) * (cols) + (j))

__global__ void madd_kernel(unsigned int *a, unsigned int *b, unsigned int *c,
 unsigned int rows, unsigned int cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = get_element_index(row, col, cols);

	c[index] = a[index] + b[index];
}
