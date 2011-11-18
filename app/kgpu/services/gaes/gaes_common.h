/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU GAES common header
 */

#ifndef __GAES_COMMON_H__
#define __GAES_COMMON_H__

struct crypto_gaes_ctr_config {
    u32 key_length;
    u32 ctr_range;
};

struct crypto_gaes_ctr_info {
    u32 key_enc[AES_MAX_KEYLENGTH_U32];
    u32 key_dec[AES_MAX_KEYLENGTH_U32];
    u32 key_length;
    u32 ctr_range;
    u8  padding[24];
    u8  ctrblk[AES_BLOCK_SIZE];	
};

#ifndef __KERNEL__
typedef struct {
    u64 a, b;
} be128;
#else
#include <crypto/b128ops.h>
#endif

#define XTS_SECTOR_SIZE 512

struct crypto_xts_info {
    u32 key_enc[AES_MAX_KEYLENGTH_U32];
    u8 padding1[16];
    u32 key_dec[AES_MAX_KEYLENGTH_U32];
    u8 padding2[12];
    u32 key_length;
    be128 ivs[XTS_SECTOR_SIZE/AES_BLOCK_SIZE];
};

#endif
