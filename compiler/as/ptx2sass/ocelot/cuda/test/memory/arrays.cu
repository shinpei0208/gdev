/*!
	\file arrays.cu

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests implementation of cudaMallocArray(), among other things

	\date Feb 12, 2010
*/

#include <stdlib.h>
#include <stdio.h>

//////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////

bool testMemcpy(bool verbose) {
	bool passed = true;
	int width = 1024, height = 512;
	int errors = 0;

	cudaChannelFormatDesc channel = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray *cuArray;
	cudaMallocArray(&cuArray, &channel, width, height);

	srand(7);

	size_t bytes = sizeof(float) * width * height;
	float *hostSource = new float[width * height];
	float *hostDest = new float[width * height];
	for (int j = 0; j < height; j++) {
		float *ptr = hostSource + j * width;
		float *dstPtr = hostDest + j * width;
		for (int i = 0; i < width; i++) {
			float x = (float)( (rand() % 1024) / 125.0f);
			ptr[i] = x;
			dstPtr[i] = -1.0f;
		}
	}

	cudaMemcpyToArray(cuArray, 0, 0, hostSource, bytes, cudaMemcpyHostToDevice);
	cudaMemcpyFromArray(hostDest, cuArray, 0, 0, bytes, cudaMemcpyDeviceToHost);

	for (int j = 0; j < height && errors < 5; j++) {
		float *srcPtr = hostSource + j * width;
		float *dstPtr = hostDest + j * width;
		for (int i = 0; i < width && errors < 5; i++) {
			float expected = srcPtr[i];
			float got = dstPtr[i];
			if (fabs(expected - got) > 0.001f) {
				++errors;
				if (verbose) {
					printf("ERROR: (%d, %d) - expected %f, got %f\n", i, j, expected, got); 
					fflush(stdout);
				}
			}
		}
	}

	cudaFreeArray(cuArray);

	delete [] hostSource;
	delete [] hostDest;

	return passed;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *arg[]) {

	bool result = testMemcpy(true);

	if (result) {
		printf("Test PASSED\n");
	}
	else {
		printf("Test FAILED\n");
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

