#include <stdio.h>

int gdev_test_openclose(void);

int main(int argc, char *argv[])
{
	if (gdev_test_openclose())
		printf("Test failed.\n");
	else
		printf("Test passed.\n");
	return 0;
}
