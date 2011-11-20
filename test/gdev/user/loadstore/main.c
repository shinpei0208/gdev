#include <stdio.h>

int gdev_test_loadstore(void);

int main(int argc, char *argv[])
{
	int ret = gdev_test_loadstore();
	if (ret < 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	return 0;
}
