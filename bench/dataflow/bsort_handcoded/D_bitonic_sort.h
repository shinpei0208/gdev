/*****************************************************************************
* Name        : D_bitonic_sort.h
* Author      : Jason Aumiller
* Version     :
* Copyright   :  
* Description : 
/*****************************************************************************/


#include "myutil.h"

#define MODULE_FILE "D_bitonic_sort_K.cubin"
#define BITONIC_BLOCK_FN "_Z16D_bitonic_sort_KPi"
#define MERGE_BLOCKS_FN "_Z16D_merge_blocks_KjjPiS_"
#define uint unsigned int

#define MAX_THREADS_PER_BLOCK 1024
#define INIT_INPUT_SIZE 64

#define SUCCESS 0
#define FAILURE -1
#define TRUE 1
#define FALSE 0
#define DEBUG 0x1
#define OUTPUT 0x2
#define GROSS_DEBUG 0x4
#define _min(x1,x2) ((x1) > (x2) ? (x2):(x1))



