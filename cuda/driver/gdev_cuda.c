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
#ifdef __KERNEL__
#include <linux/errno.h>
#else
#include <sys/errno.h>
#endif

#define SH_TEXT ".text."
#define SH_INFO ".nv.info"
#define SH_INFO_FUNC ".nv.info."
#define SH_LOCAL ".nv.local."
#define SH_SHARED ".nv.shared."
#define SH_CONST ".nv.constant"
#define SH_REL ".rel.nv.constant"
#define SH_RELSPACE ".nv.constant14"
#define SH_GLOBAL ".nv.global"
#define SH_GLOBAL_INIT ".nv.global.init"
#define NV_GLOBAL   0x10

typedef struct section_entry_ {
	uint16_t type;
	uint16_t size;
} section_entry_t;

typedef struct const_entry {
	uint32_t sym_idx;
	uint16_t base;
	uint16_t size;
} const_entry_t;

typedef struct func_entry {
	uint32_t sym_idx;
	uint32_t local_size;
} func_entry_t;

typedef struct param_entry {
	uint32_t pad; /* always -1 */
	uint16_t idx;
	uint16_t offset;
	uint32_t size;
} param_entry_t;

typedef struct stack_entry {
	uint16_t size;			
	uint16_t unk16;
	uint32_t unk32;
} stack_entry_t;

typedef struct crs_stack_size_entry {
	uint32_t size;
} crs_stack_size_entry_t;

typedef struct symbol_entry {
	uint64_t offset; /* offset in relocation (c14) */
	uint32_t unk32;
	uint32_t sym_idx;
} symbol_entry_t;

/* prototype definition. */
static int cubin_func_type
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func);

static int load_file(char **pbin, const char *fname)
{
	char *bin;
	file_t *fp;
	uint32_t len;

	if (!(fp = FOPEN(fname)))
		return -ENOENT;

	FSEEK(fp, 0, SEEK_END);
	len = FTELL(fp);
	FSEEK(fp, 0, SEEK_SET);

	if (!(bin = (char *) MALLOC(len + 1))) {
		FCLOSE(fp);
		return -ENOMEM;
	}

	if (!FREAD(bin, len, fp)) {
		FREE(bin);
		FCLOSE(fp);
		return -EIO;
	}

	FCLOSE(fp);

	*pbin = bin;

	return 0;
}

static void unload_cubin(struct CUmod_st *mod)
{
	if (mod->bin) {
		FREE(mod->bin);
		mod->bin = NULL;
	}
}

static void cubin_func_skip(char **pos, section_entry_t *e)
{
	*pos += sizeof(section_entry_t);
/*#define GDEV_DEBUG*/
#ifdef GDEV_DEBUG
	printf("/* nv.info: ignore entry type: 0x%04x, size=0x%x */\n",
		   e->type, e->size);
#ifndef __KERNEL__
	if (e->size % 4 == 0) {
		int i;
		for (i = 0; i < e->size / 4; i++) {
			uint32_t val = ((uint32_t*)*pos)[i];
			printf("0x%04x\n", val);
		}
	}
	else {
		int i;
		for (i = 0; i < e->size; i++) {
			unsigned char val = ((unsigned char*)*pos)[i];
			printf("0x%02x\n", (uint32_t)val);
		}
	}
#endif
#endif
	*pos += e->size;
}

static void cubin_func_unknown(char **pos, section_entry_t *e)
{
	GDEV_PRINT("/* nv.info: unknown entry type: 0x%.4x, size=0x%x */\n",
			   e->type, e->size);
	cubin_func_skip(pos, e);
}

static int cubin_func_0a04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	const_entry_t *ce;

	*pos += sizeof(section_entry_t);
	ce = (const_entry_t *)*pos;
	raw_func->param_base = ce->base;
	raw_func->param_size = ce->size;
	*pos += e->size;

	return 0;
}

static int cubin_func_0c04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	*pos += sizeof(section_entry_t);
	/* e->size is a parameter size, but how can we use it here? */
	*pos += e->size;

	return 0;
}

static int cubin_func_0d04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	stack_entry_t *se;

	*pos += sizeof(section_entry_t);
	se = (stack_entry_t*) *pos;
	raw_func->stack_depth = se->size;
	/* what is se->unk16 and se->unk32... */

	*pos += e->size;

	return 0;
}

static int cubin_func_1704
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	param_entry_t *pe;
	struct gdev_cuda_param *param_data;

	*pos += sizeof(section_entry_t);
	pe = (param_entry_t *)*pos;

	param_data = (struct gdev_cuda_param *)MALLOC(sizeof(*param_data));
	param_data->idx = pe->idx;
	param_data->offset = pe->offset;
	param_data->size = pe->size >> 18;
	param_data->flags = pe->size & 0x2ffff;
	
	/* append to the head of the parameter data list. */
	param_data->next = raw_func->param_data;
	raw_func->param_data = param_data;

	*pos += e->size;

	return 0;
}

static int cubin_func_1903
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	int ret;
	char *pos2;

	*pos += sizeof(section_entry_t);
	pos2 = *pos;

	/* obtain parameters information. is this really safe? */
	do {
		section_entry_t *sh_e = (section_entry_t *)pos2;
		ret = cubin_func_1704(&pos2, sh_e, raw_func);
		if (ret)
			return ret;
		raw_func->param_count++;
	} while (((section_entry_t *)pos2)->type == 0x1704);

	/* just check if the parameter size matches. */
	if (raw_func->param_size != e->size) {
		if (e->type == 0x1803) { /* sm_13 needs to set param_size here. */
			raw_func->param_size = e->size;
		}
		else {
			GDEV_PRINT("Parameter size mismatched\n");
			GDEV_PRINT("0x%x and 0x%x\n", raw_func->param_size, e->size);
		}
	}

	*pos = pos2; /* need to check if this is correct! */

	return 0;
}

static int cubin_func_1e04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	raw_func->stack_size = crse->size << 4;

	*pos += e->size;

	return 0;
}

static int cubin_func_type
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	switch (e->type) {
	case 0x0204: /* textures */
		cubin_func_skip(pos, e);
		break;
	case 0x0a04: /* kernel parameters base and size */
		return cubin_func_0a04(pos, e, raw_func);
	case 0x0b04: /* 4-byte align data relevant to params (sm_13) */
	case 0x0c04: /* 4-byte align data relevant to params (sm_20) */
		return cubin_func_0c04(pos, e, raw_func);
	case 0x0d04: /* stack information, hmm... */
		return cubin_func_0d04(pos, e, raw_func);
	case 0x1104: /* ignore recursive call */
		cubin_func_skip(pos, e);
		break;
	case 0x1204: /* some counters but what is this? */
		cubin_func_skip(pos, e);
		break;
	case 0x1803: /* kernel parameters itself (sm_13) */
	case 0x1903: /* kernel parameters itself (sm_20/sm_30) */
		return cubin_func_1903(pos, e, raw_func);
	case 0x1704: /* each parameter information */
		return cubin_func_1704(pos, e, raw_func);
	case 0x1e04: /* crs stack size information */
		return cubin_func_1e04(pos, e, raw_func);
	case 0x0001: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0x080d: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0xf000: /* maybe just padding??? */
		*pos += 4;
		break;
	case 0xffff: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0x0020: /* ??? */
		cubin_func_skip(pos, e);
		break;
	default: /* real unknown */
		cubin_func_unknown(pos, e);
		/* return -EINVAL; */
	}

	return 0;
}

static void init_mod(struct CUmod_st *mod, char *bin)
{
	int i;

	mod->bin = bin;
	mod->func_count = 0;
	mod->symbol_count = 0;
	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		mod->cmem[i].addr = 0;
		mod->cmem[i].size = 0;
		mod->cmem[i].raw_size = 0;
		mod->cmem[i].buf = NULL;
	}
	gdev_list_init(&mod->func_list, NULL);
	gdev_list_init(&mod->symbol_list, NULL);
	mod->arch = 0;
}

static void init_kernel(struct gdev_kernel *k)
{
	int i;

	k->code_addr = 0;
	k->code_size = 0;
	k->code_pc = 0;
	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		k->cmem[i].addr = 0;
		k->cmem[i].size = 0;
		k->cmem[i].offset = 0;
	}
	k->cmem_count = 0;
	k->param_buf = NULL;
	k->param_size = 0;
	k->lmem_addr = 0;
	k->lmem_size_total = 0;
	k->lmem_size = 0;
	k->lmem_size_neg = 0;
	k->lmem_base = 0;
	k->smem_size_func = 0;
	k->smem_size = 0;
	k->smem_base = 0;
	k->warp_stack_size = 0;
	k->warp_lmem_size = 0;
	k->reg_count = 0;
	k->bar_count = 0;
	k->grid_id = 0;
	k->grid_x = 0;
	k->grid_y = 0;
	k->grid_z = 0;
	k->block_x = 0;
	k->block_y = 0;
	k->block_z = 0;
}

static void init_raw_func(struct gdev_cuda_raw_func *f)
{
	int i;

	f->name = NULL;
	f->code_buf = NULL;
	f->code_size = 0;
	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		f->cmem[i].buf = NULL;
		f->cmem[i].size = 0;
	}
	f->reg_count = 0;
	f->bar_count = 0;
	f->stack_depth = 0;
	f->stack_size = 0;
	f->shared_size = 0;
	f->param_base = 0;
	f->param_size = 0;
	f->param_count = 0;
	f->param_data = NULL;
	f->local_size = 0;
	f->local_size_neg = 0;
}

static void destroy_all_functions(struct CUmod_st *mod)
{
	struct CUfunc_st *func;
	struct gdev_cuda_raw_func *raw_func;
	struct gdev_cuda_param *param_data;
	struct gdev_list *p;
	while ((p = gdev_list_head(&mod->func_list))) {
		gdev_list_del(p);
		func = gdev_list_container(p);
		raw_func = &func->raw_func;
		while (raw_func->param_data) {
			param_data = raw_func->param_data;
			raw_func->param_data = raw_func->param_data->next;
			FREE(param_data);
		}
		FREE(raw_func->name);
		FREE(func);
	}
}

static void destroy_all_symbols(struct CUmod_st *mod)
{
	struct gdev_cuda_const_symbol *cs;
	struct gdev_list *p;
	while ((p = gdev_list_head(&mod->symbol_list))) {
		gdev_list_del(p);
		cs = gdev_list_container(p);
		FREE(cs);
	}
}

static struct CUfunc_st* lookup_func_by_name(struct CUmod_st *mod, const char *name) {
	struct CUfunc_st *func;
	gdev_list_for_each(func, &mod->func_list, list_entry) {
		if (strcmp(func->raw_func.name, name) == 0) {
			return func;
		}
	}
	return NULL;
}

static struct CUfunc_st* malloc_func_if_necessary(struct CUmod_st *mod, const char *name)
{
	struct CUfunc_st *func = NULL;
	if ((func = lookup_func_by_name(mod, name))) {
		return func;
	}

	/* We allocate and initialize func and link it to mod's linked list. */
	func = MALLOC(sizeof(*func));
	if (!func) {
		return NULL;
	}
	init_kernel(&func->kernel);
	init_raw_func(&func->raw_func);
	func->raw_func.name = STRDUP(name);

	/* insert this function to the module's function list. */
	gdev_list_init(&func->list_entry, func);
	gdev_list_add(&func->list_entry, &mod->func_list);
	mod->func_count++;
	func->mod = mod;
	return func;
}

static int load_cubin(struct CUmod_st *mod, char *bin)
{
	Elf_Ehdr *ehead;
	Elf_Shdr *sheads;
	Elf_Phdr *pheads;
	Elf_Sym *symbols, *sym;
	char *strings;
	char *shstrings;
	char *nvinfo, *nvrel, *nvglobal_init;
	uint32_t symbols_size;
	int symbols_idx, strings_idx;
	int nvinfo_idx, nvrel_idx, nvrel_const_idx,	nvglobal_idx, nvglobal_init_idx;
	symbol_entry_t *sym_entry;
	section_entry_t *se;
	void *sh;
	char *sh_name;
	char *pos;
	int i, ret = 0;

	if (memcmp(bin, "\177ELF", 4))
		return -ENOENT;

	/* initialize ELF variables. */
	ehead = (Elf_Ehdr *)bin;
	sheads = (Elf_Shdr *)(bin + ehead->e_shoff);
	pheads = (Elf_Phdr *)(bin + ehead->e_phoff);
	symbols = NULL;
	strings = NULL;
	nvinfo = NULL;
	nvrel = NULL;
	nvglobal_init = NULL;
	symbols_idx = 0;
	strings_idx = 0;
	nvinfo_idx = 0;
	nvrel_idx = 0;
	nvrel_const_idx = 0;
	nvglobal_idx = 0;
	nvglobal_init_idx = 0;
	shstrings = bin + sheads[ehead->e_shstrndx].sh_offset;

	/* seek the ELF header. */
	for (i = 0; i < ehead->e_shnum; i++) {
		sh_name = (char *)(shstrings + sheads[i].sh_name);
		sh = bin + sheads[i].sh_offset;
		/* the following are function-independent sections. */
		switch (sheads[i].sh_type) {
		case SHT_SYMTAB: /* symbol table */
			symbols_idx = i;
			symbols = (Elf_Sym *)sh;
			break;
		case SHT_STRTAB: /* string table */
			strings_idx = i;
			strings = (char *)sh;
			break;
		case SHT_REL: /* relocatable: not sure if nvcc uses it... */
			nvrel_idx = i;
			nvrel = (char *)sh;
			sscanf(sh_name, "%*s%d", &nvrel_const_idx);
			break;
		default:
			/* we never know what sections (.text.XXX, .info.XXX, etc.)
			   appears first for each function XXX... */
			if (!strncmp(sh_name, SH_TEXT, strlen(SH_TEXT))) {
				struct CUfunc_st *func = NULL;
				struct gdev_cuda_raw_func *raw_func = NULL;

				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_TEXT));
				if (!func)
					goto fail_malloc_func;

				raw_func = &func->raw_func;

				/* basic information. */
				raw_func->code_buf = bin + sheads[i].sh_offset; /* ==sh */
				raw_func->code_size = sheads[i].sh_size;
				raw_func->reg_count = (sheads[i].sh_info >> 24) & 0x3f;
				raw_func->bar_count = (sheads[i].sh_flags >> 20) & 0xf;
			}
			else if (!strncmp(sh_name, SH_CONST, strlen(SH_CONST))) {
				char fname[256] = {0};
				int x; /* cX[] */
				sscanf(sh_name, SH_CONST "%d.%s", &x, fname);
				/* global constant spaces. */
				if (strlen(fname) == 0) {
					mod->cmem[x].buf = bin + sheads[i].sh_offset;
					mod->cmem[x].raw_size = sheads[i].sh_size;
				}
				else if (x >= 0 && x < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT) {
					struct CUfunc_st *func = NULL;
					/* this function does nothing if func is already allocated. */
					func = malloc_func_if_necessary(mod, fname);
					if (!func)
						goto fail_malloc_func;
					func->raw_func.cmem[x].buf = bin + sheads[i].sh_offset;
					func->raw_func.cmem[x].size = sheads[i].sh_size;
				}
			}
			else if (!strncmp(sh_name, SH_SHARED, strlen(SH_SHARED))) {
				struct CUfunc_st *func = NULL;
				/* this function does nothing if func is already allocated. */
				func =  malloc_func_if_necessary(mod, sh_name + strlen(SH_SHARED));
				if (!func)
					goto fail_malloc_func;
				func->raw_func.shared_size = sheads[i].sh_size;
				/*
				 * int x;
				 * for (x = 0; x < raw_func->shared_size/4; x++) {
				 * 		unsigned long *data = bin + sheads[i].sh_offset;
				 *		printf("0x%x: 0x%x\n", x*4, data[x]);
				 * }
				 */
			}
			else if (!strncmp(sh_name, SH_LOCAL, strlen(SH_LOCAL))) {
				struct CUfunc_st *func = NULL;
				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_LOCAL));
				if (!func)
					goto fail_malloc_func;
				func->raw_func.local_size = sheads[i].sh_size;
				func->raw_func.local_size_neg = 0x7c0; /* FIXME */
			}
			/* NOTE: there are two types of "info" sections: 
			   1. ".nv.info.funcname"
			   2. ".nv.info"
			   ".nv.info.funcname" represents function information while 
			   ".nv.info" points to all ".nv.info.funcname" sections and
			   provide some global data information.
			   NV50 doesn't support ".nv.info" section. 
			   we also assume that ".nv.info.funcname" is an end mark. */
			else if (!strncmp(sh_name, SH_INFO_FUNC, strlen(SH_INFO_FUNC))) {
				struct CUfunc_st *func = NULL;
				struct gdev_cuda_raw_func *raw_func = NULL;
				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_INFO_FUNC));
				if (!func)
					goto fail_malloc_func;

				raw_func = &func->raw_func;

				/* look into the nv.info.@raw_func->name information. */
				pos = (char *) sh;
				while (pos < (char *) sh + sheads[i].sh_size) {
					se = (section_entry_t*) pos;
					ret = cubin_func_type(&pos, se, raw_func);
					if (ret)
						goto fail_cubin_func_type;
				}
			}
			else if (!strcmp(sh_name, SH_INFO)) {
				nvinfo_idx = i;
				nvinfo = (char *) sh;
			}
			else if (!strcmp(sh_name, SH_GLOBAL)) {
				/* symbol space size. */
				symbols_size = sheads[i].sh_size;
				nvglobal_idx = i;
			}
			else if (!strcmp(sh_name, SH_GLOBAL_INIT)) {
				nvglobal_init_idx = i;
				nvglobal_init = (char *) sh;
			}
			break;
		}
	}

	/* nv.rel... "__device__" symbols? */
	for (sym_entry = (symbol_entry_t *)nvrel; 
		 (void *)sym_entry < (void *)nvrel + sheads[nvrel_idx].sh_size;
		 sym_entry++) {
		/*
		 char *sym_name, *sh_name;
		 uint32_t size;
		 sym  = &symbols[se->sym_idx];
		 sym_name = strings + sym->st_name;
		 sh_name = strings + sheads[sym->st_shndx].sh_name;
		 size = sym->st_size;
		*/
	}

	/* symbols: __constant__ variable and built-in function names. */
	for (sym = &symbols[0]; 
		 (void *)sym < (void *)symbols + sheads[symbols_idx].sh_size; sym++) {
		 char *sym_name = strings + sym->st_name;
		 char *sh_name = shstrings + sheads[sym->st_shndx].sh_name;
		 switch (sym->st_info) {
		 case 0x0: /* ??? */
			 break;
		 case 0x2: /* ??? */
			 break;
		 case 0x3: /* ??? */
			 break;
		 case 0x1:
		 case 0x11: /* __device__/__constant__ symbols */
			 if (sym->st_shndx == nvglobal_idx) { /* __device__ */
			 }
			 else { /* __constant__ */
				 int x;
				 struct gdev_cuda_const_symbol *cs = MALLOC(sizeof(*cs));
				 if (!cs) {
					 ret = -ENOMEM;
					 goto fail_symbol;
				 }
				 sscanf(sh_name, SH_CONST"%d", &x);
				 cs->idx = x;
				 cs->name = sym_name;
				 cs->offset = sym->st_value;
				 cs->size = sym->st_size;
				 gdev_list_init(&cs->list_entry, cs);
				 gdev_list_add(&cs->list_entry, &mod->symbol_list);
				 mod->symbol_count++;
			 }
			 break;
		 case 0x12: /* function symbols */
			 break;
		 case 0x22: /* quick hack: FIXME! */
			 GDEV_PRINT("sym_name: %s\n", sym_name);
			 GDEV_PRINT("sh_name: %s\n", sh_name);
			 GDEV_PRINT("st_value: 0x%llx\n", (unsigned long long)sym->st_value);
			 GDEV_PRINT("st_size: 0x%llx\n", (unsigned long long)sym->st_size);
			 break;
		 default: /* ??? */
			 GDEV_PRINT("/* unknown symbols: 0x%x\n */", sym->st_info);
			 goto fail_symbol;
		 }
	}
	if (nvinfo) { /* >= sm_20 */
		/* parse nv.info sections. */
		pos = (char*)nvinfo;
		while (pos < nvinfo + sheads[nvinfo_idx].sh_size) {
			section_entry_t *e = (section_entry_t*) pos;
			switch (e->type) {
			case 0x0704: /* texture */
				cubin_func_skip(&pos, e);
				break;
			case 0x1104:  /* function */
				cubin_func_skip(&pos, e);
				break;
			case 0x1204: /* some counters but what is this? */
				cubin_func_skip(&pos, e);
				break;
			default:
				cubin_func_unknown(&pos, e);
				/* goto fail_function; */
			}
		}
		mod->arch = GDEV_ARCH_SM_2X;
	}
	else { /* < sm_13 */
		mod->arch = GDEV_ARCH_SM_1X;
	}

	return 0;

fail_symbol:
fail_cubin_func_type:
fail_malloc_func:
	destroy_all_functions(mod);

	return ret;
}

#ifndef __KERNEL__
static int save_ptx(char *ptx_file, const char *image)
{
	int fd;
	size_t len;

	fd = mkstemp(ptx_file);
	if (fd < 0)
		return -ENOENT;

	len = strlen(image);

	write(fd, image, len);

	close(fd);

	return 0;
}

#include <ctype.h>
#define _TARGET	".target"

static int assemble_ptx(char *cubin_file, const char *ptx_file)
{
	char buffer[256];
	int fd;
	FILE *fp;
	char *p;
	char arch[64];

	fd = mkstemp(cubin_file);
	if (fd < 0)
		return -ENOENT;

	if ((fp = fopen(ptx_file, "r")) == NULL)
		return -ENOENT;

	memset(arch, 0, sizeof(arch));

	while (!feof(fp) && !ferror(fp)) {
		fgets(buffer, sizeof(buffer), fp);
		if (strncmp(buffer, _TARGET, sizeof(_TARGET) - 1) == 0) {
			p = buffer + sizeof(_TARGET) - 1;
			if (isspace(*p++)) {
				while(isspace(*p))
					p++;
				strncpy(arch, p, sizeof(arch) - 1);
				p = arch + strlen(arch) - 1;
				while (isspace(*p)) {
					*p = '\0';
					if (p-- == arch)
						break;
				}
				break;
			}
		}
	}
	
	fclose(fp);

	if (!arch[0])
		return -ENOENT;

	snprintf(buffer, sizeof(buffer), "ptxas --gpu-name %s -o %s %s",
	         arch, cubin_file, ptx_file);

	system(buffer);

	return 0;
}

CUresult gdev_cuda_load_cubin_ptx(struct CUmod_st *mod, const char *fname)
{
	char *bin;
	int ret;
	char cubin_file[16] = "/tmp/GDEVXXXXXX";

	ret = assemble_ptx(cubin_file, fname);
	if (ret)
		goto fail_compile_ptx;

	ret = load_file(&bin, cubin_file);
	if (ret)
		goto fail_load_file;

	/* initialize module. */
	init_mod(mod, bin);

	ret = load_cubin(mod, bin);
	if (ret)
		goto fail_load_cubin;

	unlink(cubin_file);

	return CUDA_SUCCESS;

fail_load_cubin:
	unload_cubin(mod);
fail_load_file:
	unlink(cubin_file);
fail_compile_ptx:
	switch (ret) {
	case -ENOMEM:
		return CUDA_ERROR_OUT_OF_MEMORY;
	case -ENOENT:
		return CUDA_ERROR_FILE_NOT_FOUND;
	default:
		return CUDA_ERROR_UNKNOWN;
	}
}
#endif

CUresult gdev_cuda_load_cubin(struct CUmod_st *mod, const char *fname)
{
	return  gdev_cuda_load_cubin_file(mod, fname);
}

CUresult gdev_cuda_load_cubin_file(struct CUmod_st *mod, const char *fname)
{
	char *bin;
	int ret;

	ret = load_file(&bin, fname);
	if (ret)
		goto fail_load_file;

	/* initialize module. */
	init_mod(mod, bin);

	ret = load_cubin(mod, bin);
	if (ret) {
#ifdef __KERNEL__
		goto fail_load_cubin;
#else
		unload_cubin(mod);

		return gdev_cuda_load_cubin_ptx(mod, fname);
#endif
	}

	return CUDA_SUCCESS;

#ifdef __KERNEL__
fail_load_cubin:
	unload_cubin(mod);
#endif
fail_load_file:
	switch (ret) {
	case -ENOMEM:
		return CUDA_ERROR_OUT_OF_MEMORY;
	case -ENOENT:
		return CUDA_ERROR_FILE_NOT_FOUND;
	default:
		return CUDA_ERROR_UNKNOWN;
	}
}

CUresult gdev_cuda_load_cubin_image(struct CUmod_st *mod, const void *image)
{
	int ret;

	/* initialize module. */
	init_mod(mod, NULL);

	ret = load_cubin(mod, (char *)image);
	if (ret) {
#ifdef __KERNEL__
		goto fail_load_cubin;
#else
		char ptx_file[16] = "/tmp/GDEVXXXXXX";

		unload_cubin(mod);

		ret = save_ptx(ptx_file, image);
		if (ret)
			goto fail_save_ptx;

		ret = gdev_cuda_load_cubin_ptx(mod, ptx_file);

		unlink(ptx_file);

		return ret;
#endif
	}

	return CUDA_SUCCESS;

#ifdef __KERNEL__
fail_load_cubin:
	unload_cubin(mod);
#else
fail_save_ptx:
#endif
	switch (ret) {
	case -ENOMEM:
		return CUDA_ERROR_OUT_OF_MEMORY;
	case -ENOENT:
		return CUDA_ERROR_FILE_NOT_FOUND;
	default:
		return CUDA_ERROR_UNKNOWN;
	}
}

CUresult gdev_cuda_unload_cubin(struct CUmod_st *mod)
{
	/* destroy functions and constant symbols:
	   use while() instead of gdev_list_for_each(), as FREE(func) will 
	   delte the entry itself in gdev_list_for_each(). */
	destroy_all_functions(mod);
	destroy_all_symbols(mod);

	unload_cubin(mod);

	return CUDA_SUCCESS;
}

static uint64_t __round_up_pow2(uint64_t x)
{
	return x;
	if (x == 0)
		return 0;
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;

	return ++x;
}

CUresult gdev_cuda_construct_kernels
(struct CUmod_st *mod, struct gdev_cuda_info *cuda_info)
{
	struct CUfunc_st *func;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;
	uint32_t mp_count, warp_count, warp_size, stack_size, chipset;
	uint32_t cmem_size_align = 0;
	int i;
	
	mp_count = cuda_info->mp_count;
	warp_count = cuda_info->warp_count;
	warp_size = cuda_info->warp_size;
	chipset = cuda_info->chipset;
	mod->code_size = 0;
	mod->sdata_size = 0;

	if ((chipset&0xf0) >= 0xe0)
	    cmem_size_align = 0xff;

	gdev_list_for_each(func, &mod->func_list, list_entry) {
		k = &func->kernel;
		f = &func->raw_func;

		k->code_size = gdev_cuda_align_code_size(f->code_size);
		k->code_pc = 0;

		k->param_size = f->param_base + f->param_size;
		if (!(k->param_buf = MALLOC(k->param_size)))
			goto fail_malloc_param;

		/* the following c[] setup is NVIDIA's nvcc-specific. */
		k->cmem_count = GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT;
		/* c0[] is a parameter list. */
		memcpy(k->param_buf, f->cmem[0].buf, f->param_base);
		if (f->param_size > 0)
			k->cmem[0].size = gdev_cuda_align_cmem_size(f->param_size + cmem_size_align);
		else
			k->cmem[0].size = gdev_cuda_align_cmem_size(f->cmem[0].size);
		k->cmem[0].offset = 0;
		for (i = 1; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
			k->cmem[i].size = gdev_cuda_align_cmem_size(f->cmem[i].size);
			k->cmem[i].offset = 0; /* no usage. */
		}

		/* c{1,15,17}[] are something unknown... 
		   CUDA doesn't work properly without the following for some reason. */
		if (k->cmem[1].size == 0) {
			k->cmem[1].size = 0x10000;
		}
		if (k->cmem[15].size == 0) {
			k->cmem[15].size = 0x10000;
		}
		if (k->cmem[17].size == 0) {
			k->cmem[17].size = k->cmem[0].size;
		}

		/* FIXME: what is the right local memory size?
		   the blob trace says lmem_size > 0xf0 and lmem_size_neg > 0x7fc. */
		k->lmem_size = gdev_cuda_align_lmem_size(f->local_size);
		k->lmem_size_neg = gdev_cuda_align_lmem_size(f->local_size_neg);

		/* shared memory size. */
		k->smem_size_func = gdev_cuda_align_smem_size(f->shared_size);
		k->smem_size = k->smem_size_func;
	
		/* warp stack and local memory sizes. */
		if (f->stack_size) {
			stack_size = f->stack_size;
		} else {
			stack_size = f->stack_depth > 16 ? f->stack_depth : 16;
			stack_size = (stack_size / 48) * 16;
		}
		k->warp_stack_size = gdev_cuda_align_stack_size(stack_size);
		k->warp_lmem_size = 
			warp_size * (k->lmem_size + k->lmem_size_neg + k->warp_stack_size); 

		/* total local memory size. 
		   k->warp_lmem_size shouldn't be aligned at this point. */
		k->lmem_size_total = warp_count * mp_count * k->warp_lmem_size;

		/* align warp and total local memory sizes. */
		if (chipset & 0xc0) {
			k->lmem_size_total = 
				gdev_cuda_align_lmem_size_total(k->lmem_size_total);
		}
		else {
			k->lmem_size_total = __round_up_pow2(k->lmem_size_total);
		}

		k->warp_lmem_size = gdev_cuda_align_warp_size(k->warp_lmem_size);

		k->reg_count = f->reg_count;
		k->bar_count = f->bar_count;

		/* code size includes code and local constant memory sizes. */
		mod->code_size += k->code_size;
		for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++)
			mod->code_size += k->cmem[i].size;
		mod->sdata_size += k->lmem_size_total;
	}

	/* code size also includes global constant memory size. */
	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		mod->cmem[i].size = gdev_cuda_align_cmem_size(mod->cmem[i].raw_size);
		mod->code_size += mod->cmem[i].size;
	}

	return CUDA_SUCCESS;

fail_malloc_param:
	gdev_list_for_each(func, &mod->func_list, list_entry) {
		k = &func->kernel;
		if (k->param_buf)
			FREE(k->param_buf);
	}
	return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult gdev_cuda_destruct_kernels(struct CUmod_st *mod)
{
	CUresult res = CUDA_SUCCESS; 
	struct CUfunc_st *func;
	struct gdev_kernel *k;

	gdev_list_for_each(func, &mod->func_list, list_entry) {
		k = &func->kernel;
		if (k->param_buf)
			FREE(k->param_buf);
		else
			res = CUDA_ERROR_DEINITIALIZED; /* appropriate? */
	}

	return res;
}

CUresult gdev_cuda_locate_sdata(struct CUmod_st *mod)
{
	struct CUfunc_st *func;
	struct gdev_kernel *k;
	uint64_t addr = mod->sdata_addr;
	uint32_t size = mod->sdata_size;
	uint32_t offset = 0;
	
	gdev_list_for_each(func, &mod->func_list, list_entry) {
		k = &func->kernel;
		if (k->lmem_size > 0) {
			k->lmem_addr = addr + offset;
		}
		offset += k->lmem_size;
		if (offset > size)
			return CUDA_ERROR_UNKNOWN;
	}
	
	return CUDA_SUCCESS;
}

CUresult gdev_cuda_locate_code(struct CUmod_st *mod)
{
	struct CUfunc_st *func;
	struct gdev_kernel *k;
	uint64_t addr = mod->code_addr;
	uint32_t size = mod->code_size;
	uint32_t offset = 0;
	int i;

	/* we locate global constant memory at the beginning so that we know
	   its address before locating local constant memory. */
	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		if (mod->cmem[i].size > 0) {
			mod->cmem[i].addr = addr + offset;
			offset += mod->cmem[i].size;
		}
	}

	gdev_list_for_each(func, &mod->func_list, list_entry) {
		k = &func->kernel;
		if (k->code_size > 0) {
			k->code_addr = addr + offset;
			offset += k->code_size;
		}
		if (offset > size)
			return CUDA_ERROR_UNKNOWN;
		for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
			if (k->cmem[i].size > 0) {
				k->cmem[i].addr = addr + offset;
				offset += k->cmem[i].size;
			}
			else if (mod->cmem[i].size > 0) {
				k->cmem[i].addr = mod->cmem[i].addr;
				k->cmem[i].size = mod->cmem[i].size;
			}
			if (offset > size)
				return CUDA_ERROR_UNKNOWN;
		}
	}
	
	return CUDA_SUCCESS;
}

CUresult gdev_cuda_memcpy_code(struct CUmod_st *mod, void *buf)
{
	struct CUfunc_st *func;
	struct gdev_kernel *k;
	struct gdev_cuda_raw_func *f;
	uint64_t addr = mod->code_addr;
	uint32_t offset;
	int i;

	for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		if (mod->cmem[i].buf) {
			offset = mod->cmem[i].addr - addr;
			memcpy(buf + offset, mod->cmem[i].buf, mod->cmem[i].raw_size);
		}
	}

	gdev_list_for_each(func, &mod->func_list, list_entry) {
		k = &func->kernel;
		f = &func->raw_func;
		if (f->code_buf) {
			offset = k->code_addr - addr;
			memcpy(buf + offset, f->code_buf, f->code_size);
		}
		for (i = 0; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
			if (f->cmem[i].buf) {
				offset = k->cmem[i].addr - addr;
				memcpy(buf + offset, f->cmem[i].buf, f->cmem[i].size);
			}
		}
	}
	
	return CUDA_SUCCESS;
}

CUresult gdev_cuda_search_function
(struct CUfunc_st **pptr, struct CUmod_st *mod, const char *name)
{
	struct CUfunc_st *func;

	gdev_list_for_each(func, &mod->func_list, list_entry) {
		if (strcmp(func->raw_func.name, name) == 0) {
			*pptr = func;
			return CUDA_SUCCESS;
		}
	}

	return CUDA_ERROR_NOT_FOUND;
}

CUresult gdev_cuda_search_symbol
(uint64_t *addr, uint32_t *size, struct CUmod_st *mod, const char *name)
{
	struct gdev_cuda_const_symbol *cs;

	gdev_list_for_each(cs, &mod->symbol_list, list_entry) {
		if (strcmp(cs->name, name) == 0) {
			*addr = mod->cmem[cs->idx].addr + cs->offset;
			*size = cs->size;
			return CUDA_SUCCESS;
		}
	}

	return CUDA_ERROR_NOT_FOUND;
}
