//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments
//    2011.12 Shinpei Kato
//        --modified to use Driver API

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include "util.h"
#include "srad.h"

#include "graphics.c"
#include "resize.c"
#include "timer.c"

/* (in library path specified to compiler)	needed by for device functions */
#include "device.c"	

CUresult extract_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &d_I};

	res = cuModuleGetFunction(&f, mod, "_Z7extractlPf");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(extract) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(extract) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult prepare_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, CUdeviceptr d_I,
 CUdeviceptr d_sums, CUdeviceptr d_sums2)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &d_I, &d_sums, &d_sums2};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z7preparelPfS_S_");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(prepare) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(prepare) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult reduce_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, int no, int mul, 
 CUdeviceptr d_sums, CUdeviceptr d_sums2)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &no, &mul, &d_sums, &d_sums2};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z6reduceliiPfS_");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(reduce) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(reduce) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult srad_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, fp lambda, int Nr, int Nc, 
 long Ne, CUdeviceptr d_iN, CUdeviceptr d_iS, CUdeviceptr d_jE, 
 CUdeviceptr d_jW, CUdeviceptr d_dN, CUdeviceptr d_dS, CUdeviceptr d_dE, 
 CUdeviceptr d_dW, fp q0sqr, CUdeviceptr d_c, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&lambda, &Nr, &Nc, &Ne, &d_iN, &d_iS, &d_jE, &d_jW, &d_dN,
					 &d_dS, &d_dE, &d_dW, &q0sqr, &d_c, &d_I};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z4sradfiilPiS_S_S_PfS0_S0_S0_fS0_S0_");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(srad) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(srad) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult srad2_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, fp lambda, int Nr, int Nc, 
 long Ne, CUdeviceptr d_iN, CUdeviceptr d_iS, CUdeviceptr d_jE,  
 CUdeviceptr d_jW, CUdeviceptr d_dN, CUdeviceptr d_dS, CUdeviceptr d_dE, 
 CUdeviceptr d_dW, CUdeviceptr d_c, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&lambda, &Nr, &Nc, &Ne, &d_iN, &d_iS, &d_jE, &d_jW, &d_dN,
					 &d_dS, &d_dE, &d_dW, &d_c, &d_I};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z5srad2fiilPiS_S_S_PfS0_S0_S0_S0_S0_");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(srad2) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(srad2) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult compress_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &d_I};

	res = cuModuleGetFunction(&f, mod, "_Z8compresslPf");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(compress) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(compress) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

int main(int argc, char *argv [])
{
	/* time */
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;
	long long time11;
	long long time12;
	struct timeval tv;

	time0 = get_time();

	/* inputs image, input paramenters */
	fp* image_ori; /* originalinput image */
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

	/* inputs image, input paramenters */
	fp* image; /* input image */
	int Nr, Nc;	/* IMAGE nbr of rows/cols/elements */
	long Ne;

	/* algorithm parameters */
	int niter; /* nbr of iterations */
	fp lambda; /* update step size */

	/* size of IMAGE */
	int r1, r2, c1, c2; /* row/col coordinates of uniform ROI */
	long NeROI;	/* ROI nbr of elements */

	/* surrounding pixel indicies */
	int *iN, *iS, *jE, *jW;	
	
	/* counters */
	int iter; /* primary loop */
	long i, j; /* image row/col */

	/* memory sizes */
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	/*******************************************************
	 * 	GPU VARIABLES
	 ******************************************************/
	/* CUDA kernel execution parameters */
	int bdx, bdy;
	int x;
	int gdx, gdy;
	int gdx2, gdy2;

	/*  memory sizes */
	int mem_size; /* matrix memory size */

	/* HOST */
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	/* DEVICE */
	CUdeviceptr d_sums;	/* partial sum */
	CUdeviceptr d_sums2;
	CUdeviceptr d_iN;
	CUdeviceptr d_iS;
	CUdeviceptr d_jE;
	CUdeviceptr d_jW;
	CUdeviceptr d_dN; 
	CUdeviceptr d_dS; 
	CUdeviceptr d_dW; 
	CUdeviceptr d_dE;
	CUdeviceptr d_I; /* input IMAGE on DEVICE */
	CUdeviceptr d_c;

	CUcontext ctx;
	CUmodule mod;
	CUresult res;

	time1 = get_time();

	/*******************************************************
	 * 	GET INPUT PARAMETERS
	 ******************************************************/
	if(argc != 5) {
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else {
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);	// it is 502 in the original image
		Nc = atoi(argv[4]);	// it is 458 in the original image
	}

	time2 = get_time();

	/*******************************************************
	 * READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	 ******************************************************/
	/* read image */
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*) malloc(sizeof(fp) * image_ori_elem);

	read_graphics("../../../data/srad/image.pgm",
				  image_ori, image_ori_rows, image_ori_cols, 1);

	time3 = get_time();

	/*******************************************************
	 * RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	 ******************************************************/
	Ne = Nr * Nc;

	image = (fp*) malloc(sizeof(fp) * Ne);

	resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

	time4 = get_time();

	/*******************************************************
	 * SETUP
	 ******************************************************/
	r1 = 0;	/* top row index of ROI */
	r2 = Nr - 1; /* bottom row index of ROI */
	c1 = 0;	/* left column index of ROI */
	c2 = Nc - 1; /* right column index of ROI */

	/* ROI image size */
	NeROI = (r2 - r1 + 1) * (c2 - c1 + 1); /* # of elements in ROI, ROI size */

	/* allocate variables for surrounding pixels */
	mem_size_i = sizeof(int) * Nr;
	iN = (int *) malloc(mem_size_i); /* north surrounding element */
	iS = (int *) malloc(mem_size_i); /* south surrounding element */
	mem_size_j = sizeof(int) * Nc;
	jW = (int *) malloc(mem_size_j); /* west surrounding element */
	jE = (int *) malloc(mem_size_j); /* east surrounding element */

	/* N/S/W/E indices of surrounding pixels (every element of IMAGE) */
	for (i = 0; i < Nr; i++) {
		iN[i] = i - 1; /* holds index of IMAGE row above */
		iS[i] = i + 1; /* holds index of IMAGE row below */
	}
	for (j = 0; j < Nc; j++) {
		jW[j] = j - 1; /* holds index of IMAGE column on the left */
		jE[j] = j + 1; /* holds index of IMAGE column on the right */
	}

	/* N/S/W/E boundary conditions, 
	   fix surrounding indices outside boundary of image */
	iN[0] = 0; /* changes IMAGE top row index from -1 to 0 */
	iS[Nr - 1] = Nr - 1; /* changes IMAGE bottom row index from Nr to Nr-1 */
	jW[0] = 0; /* changes IMAGE leftmost column index from -1 to 0 */
	jE[Nc - 1] = Nc - 1; /* changes IMAGE rightmost col idx from Nc to Nc-1 */

	/*******************************************************
	 * GPU SETUP
	 ******************************************************/

	/* call our common CUDA initialization utility function. */
	res = cuda_driver_api_init(&ctx, &mod, "./srad.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for entire IMAGE on DEVICE */
	mem_size = sizeof(fp) * Ne;	/* size of input IMAGE */
	res = cuMemAlloc(&d_I, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for coordinates on DEVICE */
	res = cuMemAlloc(&d_iN, mem_size_i);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_iS, mem_size_i);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_jE, mem_size_j);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_jW, mem_size_j);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for partial sums on DEVICE */
	res = cuMemAlloc(&d_sums, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_sums2, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for derivatives */
	res = cuMemAlloc(&d_dN, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_dS, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_dW, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_dE, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for coefficient on DEVICE */
	res = cuMemAlloc(&d_c, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/******************************************************
	 * measurement start!
	 ******************************************************/
	time_measure_start(&tv);

	/*******************************************************
	 * COPY DATA TO DEVICE 
	 ******************************************************/
	
	res = cuMemcpyHtoD(d_iN, iN, mem_size_i);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_iS, iS, mem_size_i);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_jE, jE, mem_size_j);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_jW, jW, mem_size_j);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

	/* checkCUDAError("setup"); */

	/*******************************************************
	 * KERNEL EXECUTION PARAMETERS
	 ******************************************************/

	/* all kernels operating on entire matrix */
	bdx = NUMBER_THREADS; /* define # of threads in the block */
	bdy = 1;
	x = Ne / bdx;
	/* compensate for division remainder above by adding one grid */
	if (Ne % bdx != 0) {
		x = x + 1;
	}
	gdx = x; /* define # of blocks in the grid */
	gdy = 1;

	time5 = get_time();

	/*******************************************************
	 * COPY INPUT TO GPU
	 ******************************************************/

	res = cuMemcpyHtoD(d_I, image, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

	time6 = get_time();

	/*******************************************************
	 * SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	 ******************************************************/

	res = extract_launch(mod, gdx, gdy, bdx, bdy, Ne, d_I);
	if (res != CUDA_SUCCESS) {
		printf("extract_launch failed: res = %u\n", res);
		return -1;
	}

	/* checkCUDAError("extract"); */

	time7 = get_time();

	/*******************************************************
	 * COMPUTATION
	 ******************************************************/
	/* execute main loop:
	   do for # of iterations input parameter */
	for (iter = 0; iter < niter; iter++) {
		/* printf("%d ", iter); */
		/* fflush(NULL); */

		/* execute square kernel */
		res = prepare_launch(mod, gdx, gdy, bdx, bdy, Ne, d_I, d_sums, d_sums2);
		if (res != CUDA_SUCCESS) {
			printf("prepare_launch failed: res = %u\n", res);
			return -1;
		}

		/* checkCUDAError("prepare"); */

		/* performs subsequent reductions of sums */
		gdx2 = gdx; /* original number of blocks */
		gdy2 = gdy;
		no = Ne; /* original number of sum elements */
		mul = 1; /* original multiplier */

		while (gdx2 != 0) {
			/* checkCUDAError("before reduce"); */
			/* run kernel */
			res = reduce_launch(mod, gdx2, gdy2, bdx, bdy, Ne, no, mul, 
								d_sums, d_sums2);
			if (res != CUDA_SUCCESS) {
				printf("reduce_launch failed: res = %u\n", res);
				return -1;
			}
			
			/* checkCUDAError("reduce"); */

			/* update execution parameters */
			no = gdx2; /* get current number of elements */
			if (gdx2 == 1) {
				gdx2 = 0;
			}
			else {
				mul = mul * NUMBER_THREADS;	/* update the increment */
				x = gdx2 / bdx;	/* # of blocks */
				/* compensate for division remainder above by adding one grid */
				if (gdx2 % bdx != 0) {
					x = x + 1;
				}
				gdx2 = x;
				gdy2 = 1;
			}
			/* checkCUDAError("after reduce"); */

		}

		/* checkCUDAError("before copy sum"); */

		/* copy total sums to HOST */
		mem_size_single = sizeof(fp) * 1;
		res = cuMemcpyDtoH(&total, d_sums, mem_size_single);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyDtoH failed: res = %u\n", res);
			return -1;
		}
		res = cuMemcpyDtoH(&total2, d_sums2, mem_size_single);
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyDtoH failed: res = %u\n", res);
			return -1;
		}

		/* checkCUDAError("copy sum"); */
		/* calculate statistics */
		meanROI	= total / (fp)(NeROI); /* mean (avg.) value of element in ROI */
		meanROI2 = meanROI * meanROI;
		varROI = (total2 / (fp)(NeROI)) - meanROI2; /* variance of ROI */
		q0sqr = varROI / meanROI2; /* standard deviation of ROI */
		/* execute srad kernel */
		res = srad_launch(mod, gdx, gdy, bdx, bdy,
						  lambda, // SRAD coefficient 
						  Nr, // # of rows in input image
						  Nc, // # of columns in input image
						  Ne, // # of elements in input image
						  d_iN,	// indices of North surrounding pixels
						  d_iS, // indices of South surrounding pixels
						  d_jE, // indices of East surrounding pixels
						  d_jW,	// indices of West surrounding pixels
						  d_dN,	// North derivative
						  d_dS,	// South derivative
						  d_dE,	// East derivative
						  d_dW,	// West derivative
						  q0sqr, // standard deviation of ROI 
						  d_c, // diffusion coefficient
						  d_I // output image
			);
		if (res != CUDA_SUCCESS) {
			printf("srad_launch failed: res = %u\n", res);
			return -1;
		}
		/* checkCUDAError("srad"); */
		
		/* execute srad2 kernel */
		res = srad2_launch(mod, gdx, gdy, bdx, bdy,
						   lambda,	// SRAD coefficient 
						   Nr, // # of rows in input image
						   Nc, // # of columns in input image
						   Ne, // # of elements in input image
						   d_iN, // indices of North surrounding pixels
						   d_iS, // indices of South surrounding pixels
						   d_jE, // indices of East surrounding pixels
						   d_jW, // indices of West surrounding pixels
						   d_dN, // North derivative
						   d_dS, // South derivative
						   d_dE, // East derivative
						   d_dW, // West derivative
						   d_c, // diffusion coefficient
						   d_I // output image
			);
		if (res != CUDA_SUCCESS) {
			printf("srad2_launch failed: res = %u\n", res);
			return -1;
		}
		/* checkCUDAError("srad2"); */
	}

	/* printf("\n"); */

	time8 = get_time();

	/*******************************************************
	 * SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	 ******************************************************/

	res = compress_launch(mod, gdx, gdy, bdx, bdy, Ne, d_I);
	if (res != CUDA_SUCCESS) {
		printf("compress_launch failed: res = %u\n", res);
		return -1;
	}
	/* checkCUDAError("compress"); */

	time9 = get_time();

	/*******************************************************
	 * COPY RESULTS BACK TO CPU
	 ******************************************************/

	res = cuMemcpyDtoH(image, d_I, mem_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}
	
	/* checkCUDAError("copy back"); */

	time10 = get_time();

	/*******************************************************
	 * measurement end! will print out the time.
	 ******************************************************/
	time_measure_end(&tv);

	/*******************************************************
	 * WRITE IMAGE AFTER PROCESSING
	 ******************************************************/
	write_graphics("image_out.pgm", image, Nr, Nc, 1, 255);

	time11 = get_time();

	/*******************************************************
	 * DEALLOCATE
	 ******************************************************/
	free(image_ori);
	free(image);
	free(iN); 
	free(iS); 
	free(jW); 
	free(jE);

	cuCtxSynchronize();

	cuMemFree(d_I);
	cuMemFree(d_c);
	cuMemFree(d_iN);
	cuMemFree(d_iS);
	cuMemFree(d_jE);
	cuMemFree(d_jW);
	cuMemFree(d_dN);
	cuMemFree(d_dS);
	cuMemFree(d_dE);
	cuMemFree(d_dW);
	cuMemFree(d_sums);
	cuMemFree(d_sums2);

	time12 = get_time();

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
		return -1;
	}
	
	/*******************************************************
	 * DISPLAY TIMING
	 ******************************************************/
	printf("Time spent in different stages of the application:\n");
	printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
		   (float) (time1-time0) / 1000000, 
		   (float) (time1-time0) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
		   (float) (time2-time1) / 1000000, 
		   (float) (time2-time1) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
		   (float) (time3-time2) / 1000000, 
		   (float) (time3-time2) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n",
		   (float) (time4-time3) / 1000000, 
		   (float) (time4-time3) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n", 
		   (float) (time5-time4) / 1000000, 
		   (float) (time5-time4) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
		   (float) (time6-time5) / 1000000,
		   (float) (time6-time5) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n",
		   (float) (time7-time6) / 1000000,
		   (float) (time7-time6) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COMPUTE\n",
		   (float) (time8-time7) / 1000000,
		   (float) (time8-time7) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n",
		   (float) (time9-time8) / 1000000,
		   (float) (time9-time8) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n",
		   (float) (time10-time9) / 1000000,
		   (float) (time10-time9) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n",
		   (float) (time11-time10) / 1000000,
		   (float) (time11-time10) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f %% : FREE MEMORY\n",
		   (float) (time12-time11) / 1000000,
		   (float) (time12-time11) / (float) (time12-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n",	(float) (time12-time0) / 1000000);

	return 0;
}
