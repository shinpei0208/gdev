#include <stdio.h>

int cuda_test_madd(unsigned int n, char *path);

int main(int argc, char *argv[])
{
	int rc;
	unsigned int n = 3;
	
	if (argc > 1)
		n = atoi(argv[1]);

	rc = cuda_test_madd(n, ".");
	if ( rc != 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return rc;

}
