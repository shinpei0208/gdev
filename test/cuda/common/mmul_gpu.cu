
extern "C" __global__ void multiply(unsigned int *a, unsigned int *b, unsigned int *c,
 int n)
{
	unsigned int i;
	unsigned int product = 0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < n && col < n){
	    for (i = 0; i < n; i++)
		product += a[row * n + i] * b[i * n + col];

	    c[row*n + col] = product;
	}
}

