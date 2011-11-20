#include <stdio.h>

int cuda_test_matrixmul(unsigned int n);

int main(int argc, char *argv[])
{
	unsigned int n = 3;

	if (argc > 1)
		n = atoi(argv[1]);

	if (cuda_test_matrixmul(n) < 0)
		printf("Test failed");
	else
		printf("Test passed\n");
	
	return 0;
}
