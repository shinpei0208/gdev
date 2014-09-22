/*!
	\file malloc.cu
	\author Andrew Kerr
	\brief verifies all cudaMalloc*() functions and demonstrates ability to memcpy to and from
		allocations
*/

#include <stdlib.h>
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

static int test_malloc(bool verbose) {
	int errors = 0;
	const int width = 256;
	const int height = 128;
	
	float *block = 0;

	printf("test_malloc(%d, %d)\n", width, height);
	
	size_t bytes = width * height * sizeof(float);
	
	float *host = (float *)malloc(bytes);
	if (cudaMalloc((void **)&block, bytes) != cudaSuccess) {
		printf("cudaMalloc() 0 - failed to allocate %d bytes\n", (int)bytes);
		return ++errors;
	}
	
	for (int j = 0; j < height; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width; i++) {
			ptr[i] = (float)( (i + j * width) % 128 ) / 128.0f;
		}
	}
	
	if (cudaMemcpy(block, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy() - failed to copy %d bytes to block\n", (int)bytes);
		cudaFree(block);
		free(host);
		return ++errors;
	}
	
	for (int i = 0; i < width * height; i++) {
		host[i] = -1;
	}
	
	if (cudaMemcpy(host, block, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("cudaMemcpy() - failed to copy %d bytes from block\n", (int)bytes);
		cudaFree(block);
		free(host);
		return ++errors;
	}
	
	for (int j = 0; j < height && errors < 10; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width && errors < 10; i++) {
			float expected =(float)( (i + j * width) % 128 ) / 128.0f;
			float got = ptr[i];
			if (fabs(expected - got) > 0.001f) {
				printf("ERROR 1: (%d, %d) - expected: %f, got: %f\n",
					i, j, expected, got);
				++errors;
			}
		}
	}
	
	if (errors) {
		cudaFree(block);
		free(host);
		return errors;	
	}
	
	// now use cudaMemcpy() to select individual elements by offsets and reverify
	for (int j = 0; j < height && errors < 5; j++) {
		for (int i = 0; i < width && errors < 5; i++) {
			float x = -1;
			float expected = (float)( (i + j * width) % 128 ) / 128.0f;
			if (cudaMemcpy(&x, block + i + j * width, sizeof(float),
				cudaMemcpyDeviceToHost) != cudaSuccess) {

				printf("FAILED to cudaMemcpy() on element (%d, %d)\n", i, j);
				cudaFree(block);
				free(host);
				printf("FAILED\n");
				return errors;
			}
			if (fabs(x - expected) > 0.0001f) {
				++errors;
				printf("ERROR 2: (%d, %d) - expected: %f, got: %f\n", i, j, expected, x);
				if (errors > 10) {
					cudaFree(block);
					free(host);
					printf("FAILED\n");
					return errors;
				}
			}
		}
	}
	
	if (verbose) {
		printf("%s\n", (errors ? "FAILED" : "PASSED"));
		fflush(stdout);
	}
	
	cudaFree(block);
	free(host);
	
	return errors;
}

static int test_mallocArray(bool verbose) {
	int errors = 0;
	struct cudaArray *array = 0;
	struct cudaArray *array2 = 0;
	
	const int width = 256;
	const int height = 128;
	size_t bytes = width * height * sizeof(float);

	printf("test_mallocArray(%d, %d)\n", width, height);
	
	float *host = (float *)malloc(bytes);
	struct cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	if (cudaMallocArray(&array, &desc, width, height) != cudaSuccess) {
		printf("cudaMallocArray() 0 - failed to allocate %d bytes\n", (int)bytes);
		printf(" error: %s\n", cudaGetErrorString(cudaGetLastError()));
		return ++errors;
	}
	
	for (int j = 0; j < height; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width; i++) {
			ptr[i] = (float)( (i + j * width) % 128 ) / 128.0f;
		}
	}
	
	if (cudaMemcpyToArray(array, 0, 0, host, bytes, cudaMemcpyHostToDevice) !=
		cudaSuccess) {
		printf("cudaMemcpyToArray() - failed to copy %d bytes to array\n", (int)bytes);
		cudaFreeArray(array);
		free(host);
		return ++errors;
	}
	
	for (int i = 0; i < width * height; i++) {
		host[i] = -1;
	}
	
	if (cudaMemcpyFromArray(host, array, 0, 0, bytes, cudaMemcpyDeviceToHost) !=
		cudaSuccess) {
		printf("cudaMemcpyFromArray() - failed to copy %d bytes from array\n", (int)bytes);
		cudaFreeArray(array);
		free(host);
		return ++errors;
	}
	
	for (int j = 0; j < height && errors < 10; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width && errors < 10; i++) {
			float expected =(float)( (i + j * width) % 128 ) / 128.0f;
			float got = ptr[i];
			if (fabs(expected - got) > 0.001f) {
				printf("ERROR 1: (%d, %d) - expected: %f, got: %f\n",
					i, j, expected, got);
				++errors;
			}
		}
	}
	
	// now use cudaMemcpyFromArray() to select individual elements by offsets and reverify
	for (int j = 0; j < height && errors < 5; j++) {
		for (int i = 0; i < width && errors < 5; i++) {
			float x = -1;
			float expected = (float)( (i + j * width) % 128 ) / 128.0f;
			if (cudaMemcpyFromArray(&x, array, i*sizeof(float), j, sizeof(float), 
				cudaMemcpyDeviceToHost) != cudaSuccess) {
				printf("FAILED to memcpyFromArray(%d, %d)\n", i, j);
				cudaFreeArray(array);
				free(host);
				printf("FAILED\n");
				return -1;
			}
			if (fabs(x - expected) > 0.0001f) {
				++errors;
				printf("ERROR 0: (%d, %d) - expected: %f, got: %f\n", i, j, expected, x);
				if (errors > 10) {
					cudaFreeArray(array);
					free(host);
					printf("FAILED\n");
					return -1;
				}
			}
		}
	}
	
	if (cudaMallocArray(&array2, &desc, width*2, height*2) != cudaSuccess) {
		printf("cudaMallocArray() 1 - failed to allocate array2\n");
		cudaFreeArray(array);
		free(host);		
	}
	
	for (int i = 0; i < width * height; i++) {
		host[i] = -1;
	}
	
	if (cudaMemcpyArrayToArray(array2, width, height, array, 0, 0, bytes, 
		cudaMemcpyDeviceToDevice) != cudaSuccess) {
		printf("cudaMemcpyArrayToArray() - failed\n");
		cudaFreeArray(array2);
		cudaFreeArray(array);
		free(host);
		return ++errors;
	}
	
	if (cudaMemcpyFromArray(host, array2, width, height, bytes, cudaMemcpyDeviceToHost) !=
		cudaSuccess) {
		printf("cudaMemcpyFromArray(host, array2) - failed\n");
		cudaFreeArray(array2);
		cudaFreeArray(array);
		free(host);		
	}
	
	printf("checking results from last cudaMemcpyFromArray\n"); fflush(stdout);

	for (int j = 0; j < height && errors < 10; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width && errors < 10; i++) {
			float expected =(float)( (i + j * width) % 128 ) / 128.0f;
			float got = ptr[i];
			if (fabs(expected - got) > 0.001f) {
				printf("ERROR 2: (%d, %d) - expected: %f, got: %f\n",
					i, j, expected, got);
				++errors;
			}
		}
	}	
	
	if (verbose) {
		printf("%s\n", (errors ? "FAILED" : "PASSED"));
		fflush(stdout);
	}
	
	cudaFreeArray(array);
	free(host);
			
	return errors;
}

static int test_mallocPitch(bool verbose) {
	int errors = 0;
	
	const int width = 125;
	const int height = 128;
	size_t pitch;
	
	if (verbose) { printf("[1] mallocing pitch\n"); fflush(stdout); }
	
	float *gpu0 = 0;
	if (cudaMallocPitch((void **)&gpu0, &pitch, sizeof(float)*width, height) != cudaSuccess) {
		printf("cudaMallocPitch() 0 - failed to allocate %d x %d on device\n", width, height);
		return ++errors;
	}
	
	size_t bytes = width * height * sizeof(float);
	float *host = (float *)malloc(bytes);
	for (int j = 0; j < height; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width; i++) {
			ptr[i] = (float)(i % 128) / 64.0f + 2.0f;
		}
	}
	
	if (verbose) { printf("[2] memcpying2d\n"); fflush(stdout); }
	
	if (cudaMemcpy2D(gpu0, pitch, host, width * sizeof(float), sizeof(float)*width, height, 
		cudaMemcpyHostToDevice) != cudaSuccess) {
		
		printf("cudaMemcpy2D() 0 - failed to copy %d x %d matrix to device\n", width, height);
		free(host);
		cudaFree(gpu0);
		return ++errors;
	}
	
	for (int i = 0; i < width * height; i++) {
		host[i] = -1;
	}
	
	if (verbose) { printf("[3] memcpying\n"); fflush(stdout); }
	
	if (cudaMemcpy2D(host, sizeof(float) * width, gpu0, pitch, sizeof(float)*width, height,
		cudaMemcpyDeviceToHost) != cudaSuccess) {
		
		printf("cudaMemcpy2D() 1 - failed to copy %d x %d matrix to host\n", width, height);
		free(host);
		cudaFree(gpu0);
		return ++errors;
	}
	
	for (int j = 0; j < height && errors < 5; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width && errors < 5; i++) {
			float got = ptr[i];
			float expected = (float)(i % 128) / 64.0f + 2.0f;
			if (fabs(got - expected) > 0.001f) {
				printf("ERROR 0 (%d, %d) - expected: %f, got: %f\n", i, j, expected, got);
				++errors;
			}
		}
	}
	
	if (verbose) { printf("[4] checking for errors\n"); fflush(stdout); }
	
	if (errors) {
		cudaFree(gpu0);
		free(host);
		return ++errors;
	}
	
	if (verbose) { printf("[5] mallocing\n"); fflush(stdout); }
	
	// now copy from device to device with potentially different pitch
	float *gpu1 = 0;
	size_t pitch1 = 0;
	if (cudaMallocPitch( (void **)&gpu1, &pitch1, sizeof(float)*(width+1), height) != cudaSuccess) {
		cudaFree(gpu0);
		free(host);
		printf("cudaMallocPitch() 1 - failed to allocate on device\n");
		return ++errors;
	}
	
	if (verbose) { printf("[6] memcpying\n"); fflush(stdout); }
	
	if (cudaMemcpy2D(gpu1, pitch1, gpu0, pitch, sizeof(float)*width, height, 
		cudaMemcpyDeviceToDevice) != cudaSuccess) {
		
		cudaFree(gpu0);
		cudaFree(gpu1);
		free(host);
		printf("cudaMemcpy2D() 2 - failed to copy from buffer with pitch %d to buffer with pitch %d\n",
			(int)pitch, (int)pitch1);
		return ++errors;		
	}
	for (int i = 0; i < width * height; i++) {
		host[i] = -1;
	}
	
	if (verbose) { printf("[7] memcpying\n"); fflush(stdout); }
	
	if (cudaMemcpy2D(host, sizeof(float)*width, gpu1, pitch1, sizeof(float)*width, height, 
		cudaMemcpyDeviceToHost) != cudaSuccess) {
		
		cudaFree(gpu0);
		cudaFree(gpu1);
		free(host);
		printf("cudaMemcpy2D() 3 - failed to copy from buffer with pitch %d to buffer with pitch %d\n",
			(int)pitch, (int)(sizeof(float)*width));
		return ++errors;
	}
	
	for (int j = 0; j < height && errors < 5; j++) {
		float *ptr = &host[j * width];
		for (int i = 0; i < width && errors < 5; i++) {
			float got = ptr[i];
			float expected = (float)(i % 128) / 64.0f + 2.0f;
			if (fabs(got - expected) > 0.001f) {
				printf("ERROR 0 (%d, %d) - expected: %f, got: %f\n", i, j, expected, got);
				++errors;
			}
		}
	}
	
	if (verbose) { printf("[8] final free\n"); fflush(stdout); }
	
	cudaFree(gpu0);
	cudaFree(gpu1);
	free(host);
		
	return errors;
}

static int test_malloc2d(bool verbose) {
	int errors = 0;
	
	return errors;	
}

static int test_malloc3d(bool verbose) {
	int errors = 0;
	
	return errors;	
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *arg[]) {
	int errors = 0;
	bool verbose = true;
	
	if (!errors) {
		errors += test_malloc(verbose);
	}
	
	if (!errors) {
		errors += test_mallocArray(verbose);
	}
	if (!errors) {
		errors += test_mallocPitch(verbose);
	}
	if (!errors) {
		errors += test_malloc2d(verbose);
	}
	if (!errors) {
		errors += test_malloc3d(verbose);
	}
	
	{
		printf("Pass/Fail : %s\n", (errors ? "Fail" : "Pass"));
	}
	
	return -errors;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

