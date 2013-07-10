/*
 * Copyright (C) 2011 Shinpei Kato
 *
 * Systems Research Lab, University of California at Santa Cruz
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "cuda.h"
#include "gdev_cuda.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
	CUresult res;
	struct CUmod_st mod;
	struct CUfunc_st *func;
	struct gdev_cuda_raw_func *f;
	struct gdev_cuda_param *param_data;
	const char *fname;
	int i, j;

	if (argc != 2) {
		printf("Invalid argument\n");
		exit(1);
	}

	fname = argv[1];
	if ((res = gdev_cuda_load_cubin_file(&mod, fname)) != CUDA_SUCCESS)
		goto fail_load_cubin;

	/* code dump. */
	gdev_list_for_each(func, &mod.func_list, list_entry) {
		f = &func->raw_func;
		if (f->code_buf) {
			printf("uint32_t code_%s[] = {\n", f->name);
			for (i = 0; i < f->code_size / 4; i++) {
				printf("\t0x%08x,\n", ((uint32_t*)f->code_buf)[i]);
			}
			printf("};\n");
			printf("\n");
		}
	}

	/* local constant memory dump. */
	gdev_list_for_each(func, &mod.func_list, list_entry) {
		f = &func->raw_func;
		for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
			if (f->cmem[i].buf) {
				printf("uint32_t c%d_%s[] = {\n", i, f->name);
				for (j = 0; j < f->cmem[i].size / 4; j++) {
					printf("\t0x%08x,\n", ((uint32_t*)f->cmem[i].buf)[j]);
				}
				printf("};\n");
				printf("\n");
			}
		}
	}

	/* global constant memory dump. */
	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		if (mod.cmem[i].buf) {
			printf("uint32_t c%d[] = {\n", i);
			for (j = 0; j < mod.cmem[i].raw_size / 4; j++) {
				printf("\t0x%08x,\n", ((uint32_t*)mod.cmem[i].buf)[j]);
			}
			printf("};\n");
			printf("\n");
		}
	}

	/* type dump. */
	printf("struct gdev_cudump {\n");
	printf("\tchar *name;\n");
	printf("\tvoid *code_buf;\n");
	printf("\tuint32_t code_size;\n");
	printf("\tstruct {\n");
	printf("\t\tvoid *buf;\n");
	printf("\t\tuint32_t size;\n");
	printf("\t} cmem[%d];\n", GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT);
	printf("\tuint32_t param_base;\n");
	printf("\tuint32_t param_size;\n");
	printf("\tuint32_t param_count;\n");
	printf("\tstruct {\n");
	printf("\t\tuint32_t offset;\n");
	printf("\t\tuint32_t size;\n");
	printf("\t\tuint32_t flags;\n");
	printf("\t} *param_data;\n");
	printf("\tuint32_t *param_buf;\n");
	printf("\tuint32_t local_size;\n");
	printf("\tuint32_t local_size_neg;\n");
	printf("\tuint32_t shared_size;\n");
	printf("\tuint32_t stack_depth;\n");
	printf("\tuint32_t reg_count;\n");
	printf("\tuint32_t bar_count;\n");
	printf("};\n");
	printf("\n");

	/* struct gdev_cudump dump. */
	gdev_list_for_each(func, &mod.func_list, list_entry) {
		f = &func->raw_func;
		printf("struct gdev_cudump %s = {\n", f->name);

		printf("\t.name = \"%s\",\n", f->name);
		printf("\t.code_buf = code_%s,\n", f->name);
		printf("\t.code_size = 0x%x,\n", f->code_size);
		printf("\t.cmem = {\n");
		for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
			if (mod.cmem[i].buf) {
				printf("\t\t{c%d, 0x%x},\n", i, mod.cmem[i].raw_size);
			}
			else if (f->cmem[i].buf) {
				printf("\t\t{c%d_%s, 0x%x},\n", i, f->name, f->cmem[i].size);
			}
			else {
				printf("\t\t{NULL, 0},\n");
			}
		}
		printf("\t},\n");
		printf("\t.param_base = 0x%x,\n", f->param_base);
		printf("\t.param_size = 0x%x,\n", f->param_size);
		printf("\t.param_count = 0x%x,\n", f->param_count);
		printf("\t.param_data = {\n");
		param_data = f->param_data;
		while (param_data) {
			printf("\t\t{%d, 0x%x, 0x%x, 0x%x},\n", 
				   param_data->idx, 
				   param_data->offset, 
				   param_data->size, 
				   param_data->flags);
			param_data = param_data->next;
		}
		printf("\t},\n");
		printf("\t.param_buf = NULL /* filled in later */,\n");
		printf("\t.local_size = 0x%x,\n", f->local_size);
		printf("\t.local_size_neg = 0x%x,\n", f->local_size_neg);
		printf("\t.shared_size = 0x%x,\n", f->shared_size);
		printf("\t.stack_depth = 0x%x,\n", f->stack_depth);
		printf("\t.reg_count = 0x%x,\n", f->reg_count);
		printf("\t.bar_count = 0x%x,\n", f->bar_count);

		printf("};\n");
		printf("\n");
	}

	return 0;

fail_load_cubin:
	GDEV_PRINT("Failed to load cubin\n");
	return 0;
}
