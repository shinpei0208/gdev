/*!
	\file indirectCallDriver.cu
	
	\author Andrew Kerr <arkerr@gatech.edu>
	
	\brief demonstrates indirect function calling
*/

extern "C" __device__ __noinline__ int funcDouble(int a) {
	return a*2;
}

extern "C" __device__ __noinline__ int funcTriple(int a) {
	return a*3;
}

extern "C" __device__ __noinline__ int funcQuadruple(int a) {
	return a*4;
}
extern "C" __device__ __noinline__ int funcPentuple(int a) {
	return a*5;
}

extern "C" __global__ void kernelEntry(int *A, int b) {

	/*
	int (*filter[])(int) = {
		&funcDouble,
		&funcTriple,
		&funcQuadruple,
		&funcPentuple
	};
	*/
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int p = ((b + i) & 3);
	
	int (*filter)(int);
	if (p == 0) {
		filter = &funcDouble;
	}
	else if (p == 1) {
		filter = &funcTriple;
	}
	else if (p == 2) {
		filter = &funcQuadruple;
	}
	else if (p == 3) {
		filter = &funcPentuple;
	}

	A[i] = filter(i);
}

