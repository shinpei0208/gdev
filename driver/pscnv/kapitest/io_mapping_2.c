#include "drmP.h"
#include <linux/io-mapping.h>

void dummy(struct io_mapping *iom) {
	io_mapping_map_atomic_wc(iom, 0);
}
