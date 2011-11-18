#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int gdev_test_matrixadd(uint32_t *a, uint32_t *b, uint32_t *c, int n);

int main(int argc, char *argv[])
{
	int ret;
	int n;
	uint32_t *a, *b, *c;

	if(argc != 2) {
		printf("Invalid arguments\n");
		return 0;
	}

	n = atoi(argv[1]);
	a = malloc(n * n * sizeof(uint32_t));
	b = malloc(n * n * sizeof(uint32_t));
	c = malloc(n * n * sizeof(uint32_t));

	if (gdev_test_matrixadd(a, b, c, n))
		printf("Test failed.\n");
	else
		printf("Test passed.\n");
end:
	free(a);
	free(b);
	free(c);

	return 0;
}
