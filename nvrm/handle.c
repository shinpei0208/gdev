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
