#include "gdev_api.h"

int gdev_test_openclose(void)
{
	gdev_handle_t *handle;

	if (!(handle = gopen(0))) {
		return -1;
	}
	
	gclose(handle);
 
	return 0;
}
