#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#ifdef __cplusplus
extern "C" {
#include "/usr/local/gdev/include/cuda.h"
}
#else
#include "/usr/local/gdev/include/cuda.h"
#endif
#include "util.h" /* cuda_driver_api_{init,exit}() */
#include "nn.h"

int loadData
(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest
(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage(void);
int parseCommandline
(int argc, char *argv[], char* filename, int *r, float *lat, float *lng, 
 int *q, int *t, int *p, int *d);

CUresult euclid_launch
(CUmodule mod, CUdeviceptr d_locations, CUdeviceptr d_distances, 
 int numRecords, float lat, float lng)
{
	int nr_records = numRecords; /* integer shouldn't be 64-bit? */
	int bdx, bdy, gdx, gdy;
	void* param[] = {&d_locations, &d_distances, &nr_records, &lat, &lng};
	CUfunction f;
	CUresult res;

	bdx = 16;
	bdy = 1;
	gdx = (nr_records / bdx) + (!(nr_records % bdx) ? 0 : 1);
	gdy = 1;

	res = cuModuleGetFunction(&f, mod, "_Z6euclidP7latLongPfiff");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction failed: res = %u\n", res);
		return res;
	}

	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(euclid) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

/**
 * This program finds the k-nearest neighbors
 */
int main(int argc, char* argv[])
{
	int i = 0;
	float lat, lng;
	int quiet = 0,timing = 0, platform = 0, device = 0;
	std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount = 10;
	struct timeval tv;
	/* pointers to host memory */
	float *distances;
	/* pointers to device memory */
	CUdeviceptr d_locations;
	CUdeviceptr d_distances;
	int numRecords;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;

	/* parse command line */
	if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
						 &quiet, &timing, &platform, &device)) {
		printUsage();
		return 0;
	}
	
	numRecords = loadData(filename,records,locations);
	
	if (resultsCount > numRecords) 
		resultsCount = numRecords;
	
	/*
	  for(i=0;i<numRecords;i++)
	  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,
	  locations[i].lng);
	*/
	
	/*
	 * call our common CUDA initialization utility function.
	 */
	res = cuda_driver_api_init(&ctx, &mod, "./nn.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}
	
	/*
	 * allocate memory on host and device
	 */
	distances = (float *) malloc(sizeof(float) * numRecords);
	if (!distances) {
		printf("malloc failed\n");
		return -1;
	}
	res = cuMemAlloc(&d_locations, sizeof(LatLong) * numRecords);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_distances, sizeof(float) * numRecords);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/*
	 * measurement start!
	 */
	time_measure_start(&tv);
	
	/*
	 * transfer data from host to device
	 */
	res = cuMemcpyHtoD(d_locations, &locations[0], sizeof(LatLong) * numRecords);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

	/*
	 * execute kernel
	 */
	euclid_launch(mod, d_locations,d_distances,numRecords,lat,lng);
	cuCtxSynchronize();

	/* copy data from device memory to host memory */
	res = cuMemcpyDtoH(distances, d_distances, sizeof(float) * numRecords);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}

	/*
	 * measurement end! will print out the time.
	 */
	time_measure_end(&tv);
	  
	/* find the resultsCount least distances */
	findLowest(records,distances,numRecords,resultsCount);

	/* print out results */
	if (!quiet)
	for (i = 0; i < resultsCount; i++) {
		printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
	}

	free(distances);
	/* free memory */
	cuMemFree(d_locations);
	cuMemFree(d_distances);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
		return -1;
	}

	return 0;
}

int loadData
(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations)
{
	FILE *flist,*fp;
	int	i = 0;
	char dbname[64];
	int recNum=0;
	
	/**Main processing **/
	
	flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		 * Read in all records of length REC_LENGTH
		 * If this is the last file in the filelist, then done
		 * else open next file to be read next iteration
		 */
		if(fscanf(flist, "%s\n", dbname) != 1) {
			fprintf(stderr, "error reading filelist\n");
			exit(0);
		}
		fp = fopen(dbname, "r");
		if(!fp) {
			printf("error opening a db\n");
			exit(1);
		}
		// read each record
		while(!feof(fp)){
			Record record;
			LatLong latLong;
			fgets(record.recString,49,fp);
			fgetc(fp); // newline
			if (feof(fp)) break;
			
			// parse for lat and long
			char substr[6];
			
			for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
			substr[5] = '\0';
			latLong.lat = atof(substr);
			
			for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
			substr[5] = '\0';
			latLong.lng = atof(substr);
			
			locations.push_back(latLong);
			records.push_back(record);
			recNum++;
		}
		fclose(fp);
	}
	fclose(flist);
//	for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
	return recNum;
}

void findLowest
(std::vector<Record> &records,float *distances,int numRecords,int topN)
{
	int i,j;
	float val;
	int minLoc;
	Record *tempRec;
	float tempDist;
	
	for(i=0;i<topN;i++) {
		minLoc = i;
		for(j=i;j<numRecords;j++) {
			val = distances[j];
			if (val < distances[minLoc]) minLoc = j;
		}
		// swap locations and distances
		tempRec = &records[i];
		records[i] = records[minLoc];
		records[minLoc] = *tempRec;
		
		tempDist = distances[i];
		distances[i] = distances[minLoc];
		distances[minLoc] = tempDist;
		
		// add distance to the min we just found
		records[i].distance = distances[i];
	}
}

int parseCommandline
(int argc, char *argv[], char* filename,int *r,float *lat,float *lng, int *q, 
 int *t, int *p, int *d)
{
	int i;
	if (argc < 2) return 1; // error
	strncpy(filename,argv[1],100);
	char flag;
	
	for(i=1;i<argc;i++) {
	  if (argv[i][0]=='-') {// flag
		flag = argv[i][1];
		  switch (flag) {
			case 'r': // number of results
			  i++;
			  *r = atoi(argv[i]);
			  break;
			case 'l': // lat or lng
			  if (argv[i][2]=='a') {//lat
				*lat = atof(argv[i+1]);
			  }
			  else {//lng
				*lng = atof(argv[i+1]);
			  }
			  i++;
			  break;
			case 'h': // help
			  return 1;
			  break;
			case 'q': // quiet
			  *q = 1;
			  break;
			case 't': // timing
			  *t = 1;
			  break;
			case 'p': // platform
			  i++;
			  *p = atoi(argv[i]);
			  break;
			case 'd': // device
			  i++;
			  *d = atoi(argv[i]);
			  break;
		}
	  }
	}
	if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
	  return 1;
	return 0;
}

void printUsage(void)
{
	printf("Nearest Neighbor Usage\n");
	printf("\n");
	printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
	printf("\n");
	printf("example:\n");
	printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
	printf("\n");
	printf("filename	 the filename that lists the data input files\n");
	printf("-r [int]	 the number of records to return (default: 10)\n");
	printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
	printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
	printf("\n");
	printf("-h, --help   Display the help file\n");
	printf("-q		   Quiet mode. Suppress all text output.\n");
	printf("-t		   Print timing information.\n");
	printf("\n");
	printf("-p [int]	 Choose the platform (must choose both platform and device)\n");
	printf("-d [int]	 Choose the device (must choose both platform and device)\n");
	printf("\n");
	printf("\n");
	printf("Notes: 1. The filename is required as the first parameter.\n");
	printf("	   2. If you declare either the device or the platform,\n");
	printf("		  you must declare both.\n\n");
}
