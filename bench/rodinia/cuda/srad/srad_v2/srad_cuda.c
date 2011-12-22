#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "util.h"
#include "srad.h"

#include <cuda.h>

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}

CUresult srad_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, CUdeviceptr E_C, 
 CUdeviceptr W_C, CUdeviceptr N_C, CUdeviceptr S_C, CUdeviceptr J_cuda, 
 CUdeviceptr C_cuda, int cols, int rows, float q0sqr)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&E_C, &W_C, &N_C, &S_C, &J_cuda, &C_cuda, &cols, &rows, 
					 &q0sqr};

	res = cuModuleGetFunction(&f, mod, "_Z4sradPfS_S_S_S_S_iif");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(srad) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0x1800, 0, 
						 (void**) param, NULL);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(srad) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult srad2_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, CUdeviceptr E_C, 
 CUdeviceptr W_C, CUdeviceptr N_C, CUdeviceptr S_C, CUdeviceptr J_cuda, 
 CUdeviceptr C_cuda, int cols, int rows, float lambda, float q0sqr)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&E_C, &W_C, &N_C, &S_C, &J_cuda, &C_cuda, &cols, &rows, 
					 &lambda, &q0sqr};

	res = cuModuleGetFunction(&f, mod, "_Z5srad2PfS_S_S_S_S_iiff");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(srad2) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0x1400, 0, 
						 (void**) param, NULL);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(srad2) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
	int i, j, k;
	int rows, cols, size_I, size_R, niter = 10, iter;
	float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;
	struct timeval tv;

#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
#endif

#ifdef GPU
	int gdx, gdy, bdx, bdy;
	CUdeviceptr J_cuda;
	CUdeviceptr C_cuda;
	CUdeviceptr E_C, W_C, N_C, S_C;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;
#endif

	unsigned int r1, r2, c1, c2;
	float *c;
 
	if (argc == 9) {
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
			fprintf(stderr, "rows and cols must be multiples of 16\n");
			exit(1);
		}
		r1 = atoi(argv[3]);  //y1 position of the speckle
		r2 = atoi(argv[4]);  //y2 position of the speckle
		c1 = atoi(argv[5]);  //x1 position of the speckle
		c2 = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
	}
	else {
		usage(argc, argv);
	}

	size_I = cols * rows;
	size_R = (r2 - r1 + 1) * (c2 - c1 + 1);   
	
	I = (float *) malloc(size_I * sizeof(float));
	J = (float *) malloc(size_I * sizeof(float));
	c = (float *) malloc(sizeof(float)* size_I);

#ifdef CPU
	iN = (int *) malloc(sizeof(unsigned int*) * rows);
	iS = (int *) malloc(sizeof(unsigned int*) * rows);
	jW = (int *) malloc(sizeof(unsigned int*) * cols);
	jE = (int *) malloc(sizeof(unsigned int*) * cols);	

	dN = (float *) malloc(sizeof(float)* size_I);
	dS = (float *) malloc(sizeof(float)* size_I);
	dW = (float *) malloc(sizeof(float)* size_I);
	dE = (float *) malloc(sizeof(float)* size_I);	
	

	for (i = 0; i < rows; i++) {
		iN[i] = i - 1;
		iS[i] = i + 1;
	}	
	for (j = 0; j < cols; j++) {
		jW[j] = j - 1;
		jE[j] = j + 1;
	}
	iN[0] = 0;
	iS[rows - 1] = rows - 1;
	jW[0] = 0;
	jE[cols - 1] = cols - 1;
#endif

#ifdef GPU
	/* call our common CUDA initialization utility function. */
	res = cuda_driver_api_init(&ctx, &mod, "./srad.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_init failed: res = %u\n", res);
		return;
	}

	/* Allocate device memory */
	res = cuMemAlloc(&J_cuda, sizeof(float) * size_I);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
	res = cuMemAlloc(&C_cuda, sizeof(float) * size_I);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
	res = cuMemAlloc(&E_C, sizeof(float) * size_I);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
	res = cuMemAlloc(&W_C, sizeof(float) * size_I);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
	res = cuMemAlloc(&S_C, sizeof(float) * size_I);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
	res = cuMemAlloc(& N_C, sizeof(float)* size_I);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
#endif 

	printf("Randomizing the input matrix\n");
	/* Generate a random matrix */
	random_matrix(I, rows, cols);

	for (k = 0;  k < size_I; k++) {
	 	J[k] = (float) exp(I[k]);
	}

	/******************************************************
	 * measurement start!
	 ******************************************************/
	time_measure_start(&tv);

	for (iter = 0; iter < niter; iter++) {
		sum = 0;
		sum2 = 0;
		for (i = r1; i <= r2; i++) {
			for (j = c1; j <= c2; j++) {
				tmp = J[i * cols + j];
				sum += tmp;
				sum2 += tmp * tmp;
			}
		}
		meanROI = sum / size_R;
		varROI = (sum2 / size_R) - (meanROI * meanROI);
		q0sqr = varROI / (meanROI * meanROI);
		
#ifdef CPU
		for (i = 0 ; i < rows ; i++) {
			for (j = 0; j < cols; j++) { 
				k = i * cols + j;
				Jc = J[k];
				/* directional derivates */
				dN[k] = J[iN[i] * cols + j] - Jc;
				dS[k] = J[iS[i] * cols + j] - Jc;
				dW[k] = J[i * cols + jW[j]] - Jc;
				dE[k] = J[i * cols + jE[j]] - Jc;
				
				G2 = (dN[k]*dN[k] + dS[k]*dS[k] + dW[k]*dW[k] + dE[k]*dE[k]) 
					/ (Jc*Jc);

   				L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
				den  = 1 + (.25*L);
				qsqr = num / (den*den);
 
				/* diffusion coefficent (equ 33) */
				den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr));
				c[k] = 1.0 / (1.0+den) ;
				
				/* saturate diffusion coefficent */
				if (c[k] < 0) 
					c[k] = 0;
				else if (c[k] > 1) 
					c[k] = 1;
			}
		}
		for (i = 0; i < rows; i++) {
			for (j = 0; j < cols; j++) {		
				// current index
				k = i * cols + j;
				
				// diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

				// divergence (equ 58)
				D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
				
				// image update (equ 61)
				J[k] = J[k] + 0.25*lambda*D;
			}
		}
#endif
		
#ifdef GPU
		/* Currently the input size must be divided by 16 - the block size */
		bdx = BLOCK_SIZE;
		bdy = BLOCK_SIZE;
		gdx = cols / bdx;
		gdy = rows / bdy;
		
		/* Copy data from main memory to device memory */
		res = cuMemcpyHtoD(J_cuda, J, sizeof(float) * size_I);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD failed: res = %u\n", res);
			return ;
		}
		
		/* Run kernels */
		res = srad_launch(mod, gdx, gdy, bdx, bdy, E_C, W_C, N_C, S_C, 
						  J_cuda, C_cuda, cols, rows, q0sqr); 
		res = srad2_launch(mod, gdx, gdy, bdx, bdy, E_C, W_C, N_C, S_C, 
						   J_cuda, C_cuda, cols, rows, lambda, q0sqr); 
		
		/* Copy data from device memory to main memory */
		res = cuMemcpyDtoH(J, J_cuda, sizeof(float) * size_I);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD failed: res = %u\n", res);
			return ;
		}
#endif   
	}

	/*******************************************************
	 * measurement end! will print out the time.
	 ******************************************************/
	time_measure_end(&tv);

#ifdef OUTPUT
	/* Printing output */
	printf("Printing Output:\n"); 
	for(i = 0 ; i < rows ; i++) {
		for (j = 0 ; j < cols ; j++) {
			printf("%.5f ", J[i * cols + j]); 
		}	
		printf("\n"); 
	}
#endif 

	printf("Computation Done\n");
	
	free(I);
	free(J);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
	free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU
	cuMemFree(C_cuda);
	cuMemFree(J_cuda);
	cuMemFree(E_C);
	cuMemFree(W_C);
	cuMemFree(N_C);
	cuMemFree(S_C);
	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
		return;
	}
#endif 
	free(c);
}


void random_matrix(float *I, int rows, int cols)
{
	int i, j;

	srand(7);
	
	for(i = 0 ; i < rows ; i++) {
		for (j = 0 ; j < cols ; j++) {
			I[i * cols + j] = rand() / (float) RAND_MAX;
		}
	}
}

