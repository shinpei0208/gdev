#include <cnp_runtime_api.h>
#include <iostream>

__global__ void nestedCall(int count, int* array)
{
	array[count] = count;

	if(count > 0) nestedCall<<<1, 1>>>(count - 1, array);
}

__global__ void mainKernel(int* array)
{
	nestedCall<<<1, 1>>>(10, array);
}

int main()
{
	int* array = 0;

	cudaMalloc(&array, sizeof(int) * 10);

	mainKernel<<<1, 1, 1>>>(array);

	int results[10];

	cudaMemcpy(results, array, sizeof(int) * 10, cudaMemcpyDeviceToHost);

	bool error = false;

	for(int i = 0; i < 10; ++i)
	{
		if(results[i] != i)
		{
			std::cout << "For index " << i << " expected value " << i
				<< ", but got " << results[i] << "\n";
			error = true;
		}
	}
	
	if(error)
	{
		std::cout << "Pass/Fail : Fail\n";
	}
	else
	{
		std::cout << "Pass/Fail : Pass\n";
	}

	return 0;
}


