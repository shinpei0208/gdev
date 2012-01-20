#ifndef __MADD_GPU_H
#define __MADD_GPU_H

#include <cuda.h>

#ifdef __KERNEL__
#include <linux/vmalloc.h>
#include <linux/time.h>
#define printf printk
#define malloc vmalloc
#define free vfree
#define gettimeofday(x, y) do_gettimeofday(x)
#else
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#endif

#include <gdev_api.h>

#define MODULE_FILE_NAME "madd_gpu.cubin"
#define KERNEL_NAME "_Z11madd_kernelPjS_S_jj"

/* Access an element of a matrix */
//#define get_element_index(i, j, cols) ((((i) + 1) * (cols)) + ((j) + 1))
#define get_element_index(i, j, cols) ((i) * (cols) + (j))

#define X_THREADS_PER_BLOCK 8
#define X_THREADS_PER_BLOCK_SHIFT 3
#define Y_THREADS_PER_BLOCK 8
#define Y_THREADS_PER_BLOCK_SHIFT 3

/* Contains pointers to device information from initialization */
struct device_info {
	CUdevice dev;
	CUcontext context;
	CUfunction kernel;
	CUmodule module;
};

int madd_gpu_init(struct device_info *device_info);
int madd_gpu(int a_key, int b_key, int *c_key, unsigned int rows, unsigned int
 cols);
int madd_gpu_close(struct device_info *device_info);
int shmem_device_copy(int key, int size, unsigned int *matrix, int toDevice);
int free_shmem(int key, int size);


extern unsigned int key_counter; /* global key counter for shared memory */

#endif /* __MADD_GPU_H */
