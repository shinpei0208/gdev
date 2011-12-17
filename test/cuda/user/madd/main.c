#include <stdio.h>

int cuda_test_madd(unsigned int n, char *path);

int main(int argc, char *argv[])
{
	unsigned int n = 3;

	if (argc > 1)
		n = atoi(argv[1]);

	if (cuda_test_madd(n, ".") < 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return 0;
}
