#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"
#include "lud.h"

static int do_verify = 0;

static struct option long_options[] = {
	/* name, has_arg, flag, val */
	{"input", 1, NULL, 'i'},
	{"size", 1, NULL, 's'},
	{"verify", 0, NULL, 'v'},
	{0,0,0,0}
};

int lud_cuda(CUmodule *mod, CUdeviceptr m, int matrix_dim)
{
	int i = 0;
	int bdx, bdy, gdx, gdy;
	int offset;
	int shared_size;
	float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));
	CUfunction f_diagonal, f_perimeter, f_internal;
	CUresult res;

	/* get functions. */
	res = cuModuleGetFunction(&f_diagonal, *mod, "_Z12lud_diagonalPfii");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(f_diagonal) failed\n");
		return 0;
	}
	res = cuModuleGetFunction(&f_perimeter, *mod, "_Z13lud_perimeterPfii");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(f_perimeter) failed\n");
		return 0;
	}
	res = cuModuleGetFunction(&f_internal, *mod, "_Z12lud_internalPfii");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(f_internal) failed\n");
		return 0;
	}

	/* set shared memory sizes. */
	shared_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
	res = cuFuncSetSharedSize(f_diagonal, shared_size);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetSharedSize(f_diagonal) failed\n");
		return 0;
	}
	shared_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float) * 3;
	res = cuFuncSetSharedSize(f_perimeter, shared_size);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape(f_perimeter) failed\n");
		return 0;
	}
	shared_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float) * 2;
	res = cuFuncSetSharedSize(f_internal, shared_size);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape(f_internal) failed\n");
		return 0;
	}

	/* set block sizes. */
	bdx = BLOCK_SIZE;
	bdy = 1;
	res = cuFuncSetBlockShape(f_diagonal, bdx, bdy, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape(f_diagonal) failed\n");
		return 0;
	}
	bdx = BLOCK_SIZE * 2;
	bdy = 1;
	res = cuFuncSetBlockShape(f_perimeter, bdx, bdy, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape(f_diagonal) failed\n");
		return 0;
	}
	bdx = BLOCK_SIZE;
	bdy = BLOCK_SIZE;
	res = cuFuncSetBlockShape(f_internal, bdx, bdy, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape(f_internal) failed\n");
		return 0;
	}
	
	for (i = 0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
		/* diagonal */
		offset = 0;
		cuParamSetv(f_diagonal, offset, &m, sizeof(m));
		offset += sizeof(m);
		cuParamSetv(f_diagonal, offset, &matrix_dim, sizeof(matrix_dim));
		offset += sizeof(matrix_dim);
		cuParamSetv(f_diagonal, offset, &i, sizeof(i));
		offset += sizeof(i);
        cuParamSetSize(f_diagonal, offset);
		gdx = 1;
		gdy = 1;
        res = cuLaunchGrid(f_diagonal, gdx, gdy);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchGrid(f_diagonal) failed: res = %u\n", res);
            return 0;
        }

		/* perimeter */
		offset = 0;
		cuParamSetv(f_perimeter, offset, &m, sizeof(m));
		offset += sizeof(m);
		cuParamSetv(f_perimeter, offset, &matrix_dim, sizeof(matrix_dim));
		offset += sizeof(matrix_dim);
		cuParamSetv(f_perimeter, offset, &i, sizeof(i));
		offset += sizeof(i);
        cuParamSetSize(f_perimeter, offset);
		gdx = (matrix_dim - i) / BLOCK_SIZE - 1;
		gdy = 1;
        res = cuLaunchGrid(f_perimeter, gdx, gdy);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchGrid(f_perimeter) failed: res = %u\n", res);
            return 0;
        }

		/* internal */
		offset = 0;
		cuParamSetv(f_internal, offset, &m, sizeof(m));
		offset += sizeof(m);
		cuParamSetv(f_internal, offset, &matrix_dim, sizeof(matrix_dim));
		offset += sizeof(matrix_dim);
		cuParamSetv(f_internal, offset, &i, sizeof(i));
		offset += sizeof(i);
        cuParamSetSize(f_internal, offset);
		gdx = (matrix_dim - i) / BLOCK_SIZE - 1;
		gdy = (matrix_dim - i) / BLOCK_SIZE - 1;
        res = cuLaunchGrid(f_internal, gdx, gdy);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchGrid(internal) failed: res = %u\n", res);
            return 0;
        }
	}

	/* diagonal */
	offset = 0;
	cuParamSetv(f_diagonal, offset, &m, sizeof(m));
	offset += sizeof(m);
	cuParamSetv(f_diagonal, offset, &matrix_dim, sizeof(matrix_dim));
	offset += sizeof(matrix_dim);
	cuParamSetv(f_diagonal, offset, &i, sizeof(i));
	offset += sizeof(i);
	cuParamSetSize(f_diagonal, offset);
	gdx = 1;
	gdy = 1;
	res = cuLaunchGrid(f_diagonal, gdx, gdy);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchGrid(f_diagonal) failed: res = %u\n", res);
		return 0;
	}

	free(m_debug);

	return 0;
}

int
main ( int argc, char *argv[] )
{
	int matrix_dim = 32; /* default matrix_dim */
	int opt, option_index = 0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m, *mm;
	stopwatch sw;
	char fname[256];
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUmodule module;
	CUdeviceptr d_m;
	
	while ((opt = getopt_long(argc, argv, "::vs:i:", 
							  long_options, &option_index)) != -1 ) {
		switch(opt){
        case 'i':
			input_file = optarg;
			break;
        case 'v':
			do_verify = 1;
			break;
        case 's':
			matrix_dim = atoi(optarg);
			fprintf(stderr, "Currently not supported, use -i instead\n");
			fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
			exit(EXIT_FAILURE);
        case '?':
			fprintf(stderr, "invalid option\n");
			break;
        case ':':
			fprintf(stderr, "missing argument\n");
			break;
        default:
			fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
					argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	
	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	} else {
		printf("No input file specified!\n");
		exit(EXIT_FAILURE);
	}
	
	if (do_verify){
		printf("Before LUD\n");
		print_matrix(m, matrix_dim);

		matrix_duplicate(m, &mm, matrix_dim);
	}

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	
	sprintf(fname, "./cuda/lud_cuda.cubin");
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return -1;
	}

	res = cuMemAlloc(&d_m, matrix_dim*matrix_dim*sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed\n");
		return -1;
	}

	/* beginning of timing point */
	stopwatch_start(&sw);
	res = cuMemcpyHtoD(d_m, m, matrix_dim*matrix_dim*sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	
	lud_cuda(&module, d_m, matrix_dim);
	
	res = cuMemcpyDtoH(m, d_m, matrix_dim*matrix_dim*sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	
	/* end of timing point */
	stopwatch_stop(&sw);
	printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));
	
	res = cuMemFree(d_m);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	
	res = cuModuleUnload(module);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	
	if (do_verify){
		printf("After LUD\n");
		print_matrix(m, matrix_dim);
		printf(">>>Verify<<<<\n");
		lud_verify(mm, m, matrix_dim); 
		free(mm);
	}
	
	free(m);
	
	return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
