#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include "util.h"
#include "hotspot.h"

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file)
{
	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];
	
	if( (fp = fopen(file, "w" )) == 0 )
		printf( "The file was not opened\n" );
	
	for (i=0; i < grid_rows; i++) {
		for (j=0; j < grid_cols; j++) {
			sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
			fputs(str,fp);
			index++;
		}
	}
	
	fclose(fp);	
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file)
{
  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
		printf( "The file was not opened\n" );


	for (i=0; i <= grid_rows-1; i++) {
		for (j=0; j <= grid_cols-1; j++) {
			fgets(str, STR_SIZE, fp);
			if (feof(fp))
				fatal("not enough lines in file");
			//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid file format");
			vect[i*grid_cols+j] = val;
		}
	}

	fclose(fp);	

}

/*
   compute N time steps
*/

CUresult compute_tran_temp
(CUmodule mod, CUdeviceptr MatrixPower, CUdeviceptr MatrixTemp[2], 
 int col, int row, int total_iterations, int num_iterations, int blockCols, 
 int blockRows, int borderCols, int borderRows) 
{
	int gdx = blockCols;
	int gdy = blockRows;
	int bdx = BLOCK_SIZE;
	int bdy = BLOCK_SIZE;
	
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;
	
	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);
	
	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t;
	float time_elapsed;
	time_elapsed=0.001;
	
	int src = 1, dst = 0;
	CUfunction f;
	CUresult res;

	res = cuModuleGetFunction(&f, mod, "_Z14calculate_tempiPfS_S_iiiiffffff");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction failed: res = %u\n", res);
		return 0;
	}
	
	for (t = 0; t < total_iterations; t+=num_iterations) {
		int it = MIN(num_iterations, total_iterations-t);
		int temp = src;
		src = dst;
		dst = temp;
		void *param[] = {&it, &MatrixPower, &MatrixTemp[src], &MatrixTemp[dst],
						 &col, &row, &borderCols, &borderRows, &Cap,
						 &Rx, &Ry, &Rz, &step, &time_elapsed};
		res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0xc00, 0, 
							 (void**) param, NULL);
		if (res != CUDA_SUCCESS) {
			printf("cuLaunchKernel(euclid) failed: res = %u\n", res);
			return 0;
		}
	}

	return dst;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut; 
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
	
	if (argc != 7)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||
	   (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);
	
	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];
	
    size=grid_rows*grid_cols;
	
    /* --------------- pyramid parameters --------------- */
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
	
    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));
	
    if( !FilesavingPower || !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");
	
    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n", \
		   pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);
	
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

	struct timeval tv;
    CUdeviceptr MatrixTemp[2], MatrixPower;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;
	int ret;

	/*
	 * call our common CUDA initialization utility function.
	 */
	res = cuda_driver_api_init(&ctx, &mod, "./hotspot.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_init failed: res = %u\n", res);
		return;
	}
	
    res = cuMemAlloc(&MatrixTemp[0], sizeof(float) * size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
    res = cuMemAlloc(&MatrixTemp[1], sizeof(float) * size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}
    res = cuMemAlloc(&MatrixPower, sizeof(float) * size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return;
	}

	/*
	 * measurement start!
	 */
	time_measure_start(&tv);

    res = cuMemcpyHtoD(MatrixTemp[0], FilesavingTemp, sizeof(float) * size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return;
	}
    res = cuMemcpyHtoD(MatrixPower, FilesavingPower, sizeof(float) * size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return;
	}

    ret = compute_tran_temp(mod, MatrixPower, MatrixTemp, grid_cols, grid_rows,
							total_iterations, pyramid_height, 
							blockCols, blockRows, borderCols, borderRows);

    res = cuMemcpyDtoH(MatrixOut, MatrixTemp[ret], sizeof(float) * size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", res);
		return;
	}

	/*
	 * measurement end! will print out the time.
	 */
	time_measure_end(&tv);

    writeoutput(MatrixOut, grid_rows, grid_cols, ofile);

    cuMemFree(MatrixPower);
    cuMemFree(MatrixTemp[0]);
    cuMemFree(MatrixTemp[1]);
    free(MatrixOut);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
		return;
	}
}
