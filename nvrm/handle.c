/*
 * Copyright (C) 2013 Marcin Kościelnicki <koriakin@0x04.net>
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

#include "nvrm_priv.h"
#include <stdio.h>
#include <stdlib.h>

/* just in case */
#define MAX_HANDLE 0x40000000

uint32_t nvrm_handle_alloc(struct nvrm_context *ctx) {
	struct nvrm_handle **ptr = &ctx->hchain;
	uint32_t expected = 1;
	while (*ptr && (*ptr)->handle == expected) {
		ptr = &(*ptr)->next;
		expected++;
	}
	if (expected > MAX_HANDLE) {
		fprintf(stderr, "Out of handles!\n");
		abort();
	}
	struct nvrm_handle *nhandle = malloc(sizeof *nhandle);
	nhandle->next = *ptr;
	nhandle->handle = expected;	
	*ptr = nhandle;
	return nhandle->handle;
}

void nvrm_handle_free(struct nvrm_context *ctx, uint32_t handle) {
	struct nvrm_handle **ptr = &ctx->hchain;
	while (*ptr && (*ptr)->handle != handle) {
		ptr = &(*ptr)->next;
	}
	if (!*ptr) {
		fprintf(stderr, "Tried to free nonexistent handle %08x\n", handle);
		abort();
	}
	struct nvrm_handle *phandle = *ptr;
	*ptr = phandle->next;
	free(phandle);
}
