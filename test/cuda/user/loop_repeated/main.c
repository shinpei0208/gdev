#include <stdio.h>

int cuda_test_loop(unsigned int n, int count, char *path);

int main(int argc, char *argv[])
{
	unsigned int n = 3;
	int count = 10;

	if (argc > 1)
		n = atoi(argv[1]);
	if (argc > 2)
		count = atoi(argv[2]);

	if (cuda_test_loop(n, count, ".") < 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return 0;
}
