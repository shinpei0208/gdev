/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU GAES header
 */

#ifndef __GAESK_H__
#define __GAESK_H__

#include "gaes_common.h"

#define GAES_ECB_SIZE_THRESHOLD (PAGE_SIZE-1)
#define GAES_CTR_SIZE_THRESHOLD (PAGE_SIZE-1)
#define GAES_XTS_SIZE_THRESHOLD (PAGE_SIZE-1)

long test_gaes_ecb(size_t sz, int enc);
long test_gaes_ctr(size_t sz);
long test_gaes_lctr(size_t sz);

static void cvt_endian_u32(u32* buf, int n)
{
  u8* b = (u8*)buf;
  int nb = n*4;
  
  u8 t;
  int i;
  
  for (i=0; i<nb; i+=4, b+=4) {
    t = b[0];
    b[0] = b[3];
    b[3] = t;
    
    t = b[1];
    b[1] = b[2];
    b[2] = t;
  }
}

#if 0
static void dump_page_content(u8 *p)
{
    int r,c;
    printk("dump page content:\n");
    for (r=0; r<16; r++) {
	for (c=0; c<32; c++)
	    printk("%02x ", p[r*32+c]);
	printk("\n");
    }
}

static void dump_hex(u8 *p, int r, int c)
{
    int i,j;
    printk("dump hex:\n");
    for (i=0; i<r; i++) {
	for (j=0; j<c; j++) {
	    printk("%02x ", p[c*i+j]);
	}
	printk("\n");
    }    
}
#endif /* test only */

#endif
