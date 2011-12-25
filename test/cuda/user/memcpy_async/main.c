#include <stdio.h>

int cuda_test_memcpy_async(unsigned int size);

int main(int argc, char *argv[])
{
	unsigned int size = 0x10000000; /* 256MB */

	if (argc > 1)
		sscanf(argv[1], "%x", &size);

	if (cuda_test_memcpy_async(size) < 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");

	return 0;
}
