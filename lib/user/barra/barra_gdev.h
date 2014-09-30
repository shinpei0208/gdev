/*
 * Copyright (C) 2014 Sylvain Collange <sylvain.collange@inria.fr>
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

#ifndef BARRA_GDEV_H
#define BARRA_GDEV_H



/**
 * CUDA types
 */
typedef struct CUctx_st *CUcontext;     ///< CUDA context
typedef struct CUmod_st *CUmodule;      ///< CUDA module
typedef struct CUfunc_st *CUfunction;   ///< CUDA function





void barra_dev_open();
void barra_dev_close();
int barra_get_attribute(unsigned int attrib, int dev);
CUcontext barra_ctx_new(int dev);
void barra_ctx_free(CUcontext ctx);
int64_t barra_mem_alloc(unsigned int bytesize);
void barra_mem_free(int64_t addr);
int barra_read(void *buf, uint64_t addr, uint32_t size);
int barra_write(uint64_t addr, const void *buf, uint32_t size);


#endif
