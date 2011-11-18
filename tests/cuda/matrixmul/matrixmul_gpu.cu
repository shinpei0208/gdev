__global__ void multiply
(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int *d)
{
	unsigned int n = *d;
	int id = blockIdx.x * n + threadIdx.x; // blockDim.x doesn't work...
	int row = blockIdx.x;
	int col = threadIdx.x;
	unsigned int tmp  = 0;

	for(int i = 0; i < n; i++){
		tmp += a[row*n+i] * b[col+i*n];
	}
	c[id] = tmp;
}

