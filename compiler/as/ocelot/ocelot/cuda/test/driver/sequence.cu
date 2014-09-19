/*!
	\file sequence.cu
	\brief demonstrates the world-famous CUDA sequence program
*/

extern "C" __global__ void sequence(int *A, int N) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N) {
		A[tid] = 1 + tid;
	}
}


