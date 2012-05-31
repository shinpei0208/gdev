#include <stdio.h>

int cuda_test_madd_host(unsigned int n, char *path);

int main(int argc, char *argv[])
{
	unsigned int n = 3;

	if (argc > 1)
		n = atoi(argv[1]);

	int rc = cuda_test_madd_host(n, ".");
	if ( rc != 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return rc;

}
