#include <stdio.h>

int main(int argc, char *argv[])
{
	int id = -1;
	int i, n;
	char fname[256], s[8];
	FILE *fp;

	if (argc > 1)
		id = atoi(argv[1]);

	if (id != -1) {
		for (;;) {
			sprintf(fname, "/proc/gdev/vd%d/compute_bandwidth_used", id);
			if (!(fp = fopen(fname, "r")))
				return -1;
			fgets(s, 8, fp);
			printf("Compute %03d %\n", atoi(s));
			fclose(fp);

			sprintf(fname, "/proc/gdev/vd%d/memory_bandwidth_used", id);
			if (!(fp = fopen(fname, "r")))
				return -1;
			fgets(s, 8, fp);
			printf("Memory  %03d %\n", atoi(s));
			fclose(fp);

			sleep(1);
		}
	}
	else {
		if (!(fp = fopen("/proc/gdev/virtual_device_count", "r")))
			return -1;
		fgets(s, 8, fp);
		n = atoi(s);
		fclose(fp);
		printf("        ");
		for (i = 0; i < n; i++) {
			printf("gdev%d ", i);
		}
		printf("\n");
		for (;;) {
			printf("Compute ");
			for (i = 0; i < n; i++) {
				sprintf(fname, "/proc/gdev/vd%d/compute_bandwidth_used", i);
				if (!(fp = fopen(fname, "r")))
					return -1;
				fgets(s, 8, fp);
				printf("  %03d ", atoi(s));
				fclose(fp);
			}
			printf("%\n");

			printf("Memory  ");
			for (i = 0; i < n; i++) {
				sprintf(fname, "/proc/gdev/vd%d/memory_bandwidth_used", i);
				if (!(fp = fopen(fname, "r")))
					return -1;
				fgets(s, 8, fp);
				printf("  %03d ", atoi(s));
				fclose(fp);
			}
			printf("%\n");

			sleep(1);
		}
	}
		
	return 0;
}
