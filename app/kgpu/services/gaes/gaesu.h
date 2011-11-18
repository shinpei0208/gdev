/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
 
#ifndef __G_AESU_H__
#define __G_AESU_H__

/*
 * These are copied from linux/crypto/aes.h because it is an internal
 * header that can't be included by userpace programs.
 */
typedef unsigned long long int u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define AES_MAX_KEYLENGTH	(15 * 16)
#define AES_MAX_KEYLENGTH_U32	(AES_MAX_KEYLENGTH / sizeof(u32))
#define AES_BLOCK_SIZE          (16)

#include "gaes_common.h"

struct crypto_aes_ctx {
	u32 key_enc[AES_MAX_KEYLENGTH_U32];
	u32 key_dec[AES_MAX_KEYLENGTH_U32];
	u32 key_length;
};

#define ENDIAN_SELECTOR 0x00000123

#define GETU32(plaintext) __byte_perm(*(u32*)(plaintext), 0, ENDIAN_SELECTOR)

#define PUTU32(ciphertext, st) {*(u32*)(ciphertext) = __byte_perm((st), 0, ENDIAN_SELECTOR);}
/*
#define GETU32(plaintext) (((u32)(plaintext)[0] << 24) ^ \
                    ((u32)(plaintext)[1] << 16) ^ \
                    ((u32)(plaintext)[2] <<  8) ^ \
                    ((u32)(plaintext)[3]))

#define PUTU32(ciphertext, st) { (ciphertext)[0] = (u8)((st) >> 24); \
                         (ciphertext)[1] = (u8)((st) >> 16); \
                         (ciphertext)[2] = (u8)((st) >>  8); \
                         (ciphertext)[3] = (u8)(st); }
*/                       
#endif
