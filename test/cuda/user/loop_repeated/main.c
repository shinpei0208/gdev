#include <stdio.h>

int cuda_test_loop_repeated(unsigned int n, int sec, int id, char *path);

int main(int argc, char *argv[])
{
	unsigned int n = 3;
	int sec = 10;
	int id = 0;

	if (argc > 1)
		n = atoi(argv[1]);
	if (argc > 2)
		sec = atoi(argv[2]);
	if (argc > 3)
		id = atoi(argv[3]);

	if (cuda_test_loop_repeated(n, sec, id, ".") < 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return 0;
}
