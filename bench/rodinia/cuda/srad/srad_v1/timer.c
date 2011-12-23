#include <stdlib.h>
#include <sys/time.h>

 // Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
