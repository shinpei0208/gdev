/*
 * mmult_gpu.cu -- Device code for matrix multiplication benchmark
 *
 * Michael McThrow
 */

#define get_element_index(i, j, cols) ((i) * (cols) + (j))

__global__ void mmult_kernel(unsigned int *a, unsigned int *b, unsigned int *c,
 unsigned int rows, unsigned int cols)
{
	unsigned int i;
	unsigned int product = 0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = get_element_index(row, col, cols);

	for (i = 0; i < cols; i++)
		product += a[row * cols + i] + b[i * cols + col];

	c[index] = product;
}
