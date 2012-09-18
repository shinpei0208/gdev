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
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "cuda.h"
#include "gdev_cuda.h"
#include <sys/errno.h>

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

typedef struct symbol_entry {
	uint64_t offset; /* offset in relocation (c14) */
	uint32_t unk32;
	uint32_t sym_idx;
} symbol_entry_t;

/* prototype definition. */
static int cubin_func_type
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func);

static int load_bin(char **pbin, file_t **pfp, const char *fname)
{
	char *bin;
	file_t *fp;
	uint32_t len;

	if (!(fp = FOPEN(fname)))
		return -ENOENT;

	FSEEK(fp, 0, SEEK_END);
	len = FTELL(fp);
	FSEEK(fp, 0, SEEK_SET);

	if (!(bin = (char *) MALLOC(len + 1)))
		return -ENOMEM;

	if (!FREAD(bin, len, fp)) {
		FREE(bin);
		FCLOSE(fp);
		return -EIO;
	}

	*pbin = bin;
	*pfp = fp;

	return 0;
}

static void unload_bin(char *bin, file_t *fp)
{
	FREE(bin);
	FCLOSE(fp);
}

static void cubin_func_skip(char **pos, section_entry_t *e)
{
	*pos += sizeof(section_entry_t);
/* #define GDEV_DEBUG */
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

	if (raw_func->param_count == 0 && !raw_func->param_info) {
		raw_func->param_count = ce->size / 4;
		raw_func->param_info = MALLOC(raw_func->param_count * sizeof(*raw_func->param_info));
		if (!raw_func->param_info) {
			GDEV_PRINT("/* nv.info: entry type: 0x0a04: Out of memory */\n");
			return -ENOMEM;
		}
	}

	return 0;
}

static int cubin_func_0c04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	*pos += sizeof(section_entry_t);

	if (raw_func->param_count == 0 && !raw_func->param_info) {
		raw_func->param_count = e->size / 4;
		raw_func->param_info = MALLOC(raw_func->param_count * sizeof(*raw_func->param_info));
		if (!raw_func->param_info) {
			GDEV_PRINT("/* nv.info: entry type: 0x0c04: Out of memory */\n");
			return -ENOMEM;
		}
	}

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

	*pos += sizeof(section_entry_t);
	pe = (param_entry_t *)*pos;

	/* maybe useful to check parameter format later? */
	raw_func->param_info[pe->idx].offset = pe->offset;
	raw_func->param_info[pe->idx].size = pe->size >> 18;
	raw_func->param_info[pe->idx].flags = pe->size & 0x2ffff;

	*pos += e->size;

	return 0;
}

static int cubin_func_1903
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	int i, ret;;
	char *pos2;

	*pos += sizeof(section_entry_t);
	pos2 = *pos;

	for (i = 0; i < raw_func->param_count; i++) {
		section_entry_t *sh_e = (section_entry_t *)pos2;
		ret = cubin_func_type(&pos2, sh_e, raw_func);
		if (ret)
			return ret;
	}

	/* just check if the parameter size matches. */
	if (raw_func->param_size != e->size) {
		GDEV_PRINT("Parameter size mismatched\n");
		GDEV_PRINT("0x%x and 0x%x\n", raw_func->param_size, e->size);
	}

	*pos = pos2 + e->size; /* need to check if this is correct! */

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
	case 0x0c04: /* 4-byte align data relevant to params. */
		return cubin_func_0c04(pos, e, raw_func);
	case 0x0d04: /* stack information, hmm... */
		return cubin_func_0d04(pos, e, raw_func);
	case 0x1104: /* ignore recursive call */
		cubin_func_skip(pos, e);
		break;
	case 0x1204: /* some counters but what is this? */
		cubin_func_skip(pos, e);
		break;
	case 0x1903: /* kernel parameters itself */
		return cubin_func_1903(pos, e, raw_func);
	case 0x1704: /* each parameter information */
		return cubin_func_1704(pos, e, raw_func);
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

static void init_mod(struct CUmod_st *mod, char *bin, file_t *fp)
{
	int i;

	mod->bin = bin;
	mod->fp = fp;
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
	f->shared_size = 0;
	f->param_base = 0;
	f->param_size = 0;
	f->param_count = 0;
	f->param_info = NULL;
	f->local_size = 0;
	f->local_size_neg = 0;
}

static int cubin_func
(char **pos, section_entry_t *e, Elf_Sym *symbols, Elf_Ehdr *ehead, Elf_Shdr *sheads, char *strings, char *shstrings, char *bin, struct CUmod_st *mod)
{
	int i, ret;
	int sh_text_idx;
	char *sh_text_name;
	uint32_t code_idx;
	func_entry_t *fe;
	struct CUfunc_st *func;
	struct gdev_cuda_raw_func *raw_func;

	*pos += sizeof(section_entry_t);
	fe = (func_entry_t*)*pos;
	*pos += e->size;

	/* there are some __device__ functions included, but we can just ignore 
	   them... */
	if (!(symbols[fe->sym_idx].st_other & NV_GLOBAL)) {
		return 0;
	}

	/* allocate memory for a new function. */
	if (!(func = MALLOC(sizeof(*func))))
		goto fail_malloc_func;

	init_kernel(&func->kernel);
	init_raw_func(&func->raw_func);

	raw_func = &func->raw_func;

	sh_text_idx = symbols[fe->sym_idx].st_shndx;
	sh_text_name = (char*)(bin + sheads[ehead->e_shstrndx].sh_offset + 
						   sheads[sh_text_idx].sh_name);
	code_idx = symbols[fe->sym_idx].st_shndx;

	/* function members. */
	raw_func->name = strings + symbols[fe->sym_idx].st_name;
	raw_func->code_buf = bin + sheads[code_idx].sh_offset;
	raw_func->code_size = sheads[code_idx].sh_size;
	raw_func->local_size = fe->local_size;
	raw_func->local_size_neg = 0; /* FIXME */
	raw_func->reg_count = (sheads[code_idx].sh_info >> 24) & 0x3f;
	raw_func->bar_count = (sheads[code_idx].sh_flags >> 20) & 0xf;

	for (i = 0; i < ehead->e_shnum; i++) {
		char *sh_name = (char*)(shstrings + sheads[i].sh_name);

		/* nv.shared section */
		if (strncmp(sh_name, SH_SHARED, strlen(SH_SHARED)) == 0) {
			if (strcmp(sh_name + strlen(SH_SHARED), raw_func->name) == 0) {
				raw_func->shared_size = sheads[i].sh_size;
			}
		}
		else if (strncmp(sh_name, SH_LOCAL, strlen(SH_LOCAL)) == 0) {
			if (strcmp(sh_name + strlen(SH_LOCAL), raw_func->name) == 0) {
				/* perhaps we can use .nv.info's information but meh... */
				raw_func->local_size = sheads[i].sh_size;
			}
		}
		else if (strncmp(sh_name, SH_CONST, strlen(SH_CONST)) == 0) {
			int x;
			char fname[256] = {0};

			sscanf(sh_name, SH_CONST"%d.%s", &x, fname);
			/* is there any local constant other than c0[]? */
			if (x >= 0 && x < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT) {
				if (strcmp(fname, raw_func->name) == 0) {
					raw_func->cmem[x].buf = bin + sheads[i].sh_offset;
					raw_func->cmem[x].size = sheads[i].sh_size;
				}
			}
		}
		else {
			char *sh = bin + sheads[i].sh_offset;
			char *sh_pos = sh;

			/* skip if not nv.infoXXX */
			if (strncmp(sh_name, SH_INFO, strlen(SH_INFO)) != 0)
				continue;
			/* skip if not nv.info.XXX (could be just "nv.info"). */
			if (strlen(sh_name) == strlen(SH_INFO))
				continue;
			/* skip if this is a different function's nv.info.XXX */
			if (strcmp(raw_func->name, sh_name + strlen(SH_INFO) + 1))
				continue;

			/* look into the nv.info.@raw_func->name information. */
			while (sh_pos < sh + sheads[i].sh_size) {
				section_entry_t *sh_e = (section_entry_t*)sh_pos;
				ret = cubin_func_type(&sh_pos, sh_e, raw_func);
				if (ret)
					goto fail_cubin_func;
			}
		}
	}

	gdev_list_init(&func->list_entry, func);
	gdev_list_add(&func->list_entry, &mod->func_list);
	mod->func_count++;
	func->mod = mod;

	return 0;

fail_cubin_func:
	FREE(func);
fail_malloc_func:
	return ret;
}

CUresult gdev_cuda_load_cubin(struct CUmod_st *mod, const char *fname)
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
	char *pos;
	char *bin;
	file_t *fp;
	int i, ret;

	ret = load_bin(&bin, &fp, fname);
	if (ret)
		goto fail_load_bin;

	/* initialize module. */
	init_mod(mod, bin, fp);

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
		char *name = (char *)(shstrings + sheads[i].sh_name);
		void *section = bin + sheads[i].sh_offset;
		/* the following are function-independent sections. */
		switch (sheads[i].sh_type) {
		case SHT_SYMTAB: /* symbol table */
			symbols_idx = i;
			symbols = (Elf_Sym *)section;
			break;
		case SHT_STRTAB: /* string table */
			strings_idx = i;
			strings = (char *)section;
			break;
		case SHT_REL: /* relocatable: not sure if nvcc uses it... */
			nvrel_idx = i;
			nvrel = (char *)section;
			sscanf(name, "%*s%d", &nvrel_const_idx);
			break;
		default:
			/* NOTE: there are two types of info sections, which do not match
			   each other: "nv.info.funcname" and ".nv.info". we are now at 
			   ".nv.info", but it should be scanned again to recover function
			   index, which should be used to obtain all device functions 
			   without relying on section order and naming convention. */
			if (strcmp(name, SH_INFO) == 0) {
				nvinfo_idx = i;
				nvinfo = (char *)section;
			}
			else if (strcmp(name, SH_GLOBAL) == 0) {
				/* symbol space size. */
				symbols_size = sheads[i].sh_size;
				nvglobal_idx = i;
			}
			else if (strcmp(name, SH_GLOBAL_INIT) == 0) {
				nvglobal_init_idx = i;
				nvglobal_init = (char *)section;
			}
			else if (strncmp(name, SH_CONST, strlen(SH_CONST)) == 0) {
				char func[256] = {0};
				int x; /* cX[] */
				sscanf(name, SH_CONST"%d.%s", &x, func);
				/* global constant spaces. */
				if (strlen(func) == 0) {
					mod->cmem[x].buf = bin + sheads[i].sh_offset;
					mod->cmem[x].raw_size = sheads[i].sh_size;
				}
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
		 default: /* ??? */
			 GDEV_PRINT("/* unknown symbols: 0x%x\n */", sym->st_info);
			 goto fail_symbol;
		 }
	}

	/* parse nv.info sections. */
	pos = (char*)nvinfo;
	while (pos < nvinfo + sheads[nvinfo_idx].sh_size) {
		section_entry_t *e = (section_entry_t*) pos;
		switch (e->type) {
		case 0x0704: /* texture */
			cubin_func_skip(&pos, e);
			break;
		case 0x1104:  /* function */
			ret = cubin_func(&pos, e, symbols, ehead, sheads, strings, 
							 shstrings, bin, mod);
			if (ret)
				goto fail_function;
			break;
		case 0x1204: /* some counters but what is this? */
			cubin_func_skip(&pos, e);
			break;
		default:
			cubin_func_unknown(&pos, e);
			/* goto fail_function; */
		}
	}

	return CUDA_SUCCESS;

fail_function:
fail_symbol:
	unload_bin(bin, fp);
fail_load_bin:
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
	struct CUfunc_st *func;
	struct gdev_cuda_raw_func *raw_func;
	struct gdev_cuda_const_symbol *cs;
	struct gdev_list *p;

	if (!mod->bin || !mod->fp)
		return CUDA_ERROR_INVALID_VALUE;

	/* destroy functions and constant symbols:
	   use while() instead of gdev_list_for_each(), as FREE(func) will 
	   delte the entry itself in gdev_list_for_each(). */
	while ((p = gdev_list_head(&mod->func_list))) {
		gdev_list_del(p);
		func = gdev_list_container(p);
		raw_func = &func->raw_func;
		FREE(raw_func->param_info);
		FREE(func);
	}
	while ((p = gdev_list_head(&mod->symbol_list))) {
		gdev_list_del(p);
		cs = gdev_list_container(p);
		FREE(cs);
	}

	unload_bin(mod->bin, mod->fp);

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
	uint32_t mp_count, warp_count, warp_size, chipset;
	int i;
	
	mp_count = cuda_info->mp_count;
	warp_count = cuda_info->warp_count;
	warp_size = cuda_info->warp_size;
	chipset = cuda_info->chipset;
	mod->code_size = 0;
	mod->sdata_size = 0;

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
		k->cmem[0].size = gdev_cuda_align_cmem_size(f->param_size);
		k->cmem[0].offset = 0;
		for (i = 1; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
			k->cmem[i].size = gdev_cuda_align_cmem_size(f->cmem[i].size);
			k->cmem[i].offset = 0; /* no usage. */
		}

		/* c{1,15,17}[] are something unknown... */
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
		k->smem_size = gdev_cuda_align_smem_size(f->shared_size);
	
		/* warp stack and local memory sizes. */
		/* k->warp_stack_size = f->stack_depth * 9; */
		k->warp_stack_size = gdev_cuda_align_stack_size(f->stack_depth);
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
