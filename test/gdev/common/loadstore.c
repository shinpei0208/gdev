#include "gdev_api.h"

/*
  mov b32 $r6 0xdeadcafe
  mov b32 $r8 $nphysid
  mov b32 $r10 $physid
  ld b64 $r0d c0[0]
  ld b64 $r2d c0[8]
  ld b64 $r4d c0[16]
  st b32 wb g[$r0d] $r6
  st b32 wb g[$r2d] $r8
  st b32 wb g[$r4d] $r10
  exit
*/
uint32_t kcode[] = {
	0xf8019de2,
	0x1b7ab72b,
	0x08021c04,
	0x2c000000,
	0x0c029c04,
	0x2c000000,
	0x03f01ca6,
	0x14000000,
	0x23f09ca6,
	0x14000000,
	0x43f11ca6,
	0x14000000,
	0x00019c85,
	0x94000000,
	0x00221c85,
	0x94000000,
	0x00429c85,
	0x94000000,
	0x00001de7,
	0x80000000
};

static inline unsigned __round_up_pow2(unsigned x)
{
	if (x == 0)
		return 0;
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;

	return ++x;
}

int gdev_test_loadstore(void)
{
	uint32_t mp_count;
	uint32_t stack_depth, stack_size;
	uint32_t param_buf[6 * 4];
	uint32_t code_size, data_size;
	uint32_t id;
	uint64_t data_addr;
	uint64_t result[3];

	gdev_handle_t *handle;
	struct gdev_kernel k;

	if (!(handle = gopen(0))) {
		return -1;
	}
	
	code_size = sizeof(kcode);
	if (code_size & 0xff)
		code_size = (code_size + 0x100) & ~0xff;
	k.code_pc = 0;
	k.cmem_segment = 0;
	k.cmem_size = 3 * 8; /* 3 parameters */
	if (k.cmem_size == 0 || k.cmem_size & 0xff)
		k.cmem_size = (k.cmem_size + 0x100) & ~0xff;
	k.lmem_size = 0x100; /* just random */
	if (k.lmem_size & 0xf)
		k.lmem_size = (k.lmem_size + 0x10) & ~0xf;
	k.lmem_size_neg = 0; /* just random */
	if (k.lmem_size_neg & 0xf)
		k.lmem_size_neg = (k.lmem_size_neg + 0x10) & ~0xf;
	k.lmem_base = 0x01000000;
	k.smem_size = 0x100; /* just random */
	if (k.smem_size & 0x7f)
		k.smem_size = (k.smem_size + 0x80) & (~0x7f);
	k.smem_base = 0x0;
	
	/* stack depth must be >= 16? */
	stack_depth = 16; 
	/* stack level is round_up(stack_depth/48) */
	k.stack_level = stack_depth / 48;
	if (stack_depth % 48 != 0 && stack_depth > 16)
		k.stack_level++;
	/* this is the stack size */
	stack_size = k.stack_level * 16;
	
	/* FIXME: per-thread warp size may differ from 32. */
	k.warp_size = 32 * (stack_size + k.lmem_size + k.lmem_size_neg); 
	
	/* FIXME: the number of active warps may differ from 48. */
	gquery(handle, GDEV_QUERY_NVIDIA_MP_COUNT, &mp_count);
	k.lmem_size_total = 48 * mp_count * k.warp_size;
	k.lmem_size_total = __round_up_pow2(k.lmem_size_total);
	if (k.lmem_size_total > 128 * 1024)
		k.lmem_size_total = 128 * 1024;

	if (!(k.code_addr = gmalloc(handle, code_size)))
		return -1;
	if (!(k.cmem_addr = gmalloc(handle, k.cmem_size)))
		return -1;
	if (!(k.lmem_addr = gmalloc(handle, k.lmem_size_total)))
		return -1;
	data_size = 3 * 8;
	if (!(data_addr = gmalloc(handle, data_size)))
		return -1;
	
	k.param_count = 6; /* note param is integer size. */
	k.param_buf = param_buf;
	k.param_buf[0] = data_addr;
	k.param_buf[1] = data_addr >> 32;
	k.param_buf[2] = data_addr + 8;
	k.param_buf[3] = (data_addr + 8) >> 32;
	k.param_buf[4] = data_addr + 16;
	k.param_buf[5] = (data_addr + 16) >> 32;
	k.param_start = 0;
	
	k.reg_count = 32;
	k.bar_count = 0;
	k.grid_id = 1;
	
	k.grid_x = 1;
	k.grid_y = 1;
	k.grid_z = 1;
	k.block_x = 1;
	k.block_y = 1;
	k.block_z = 1;
	
	gmemcpy_to_device(handle, k.code_addr, kcode, code_size);
	
	glaunch(handle, &k, &id);
	gsync(handle, id);
	
	gmemcpy_from_device(handle, result, data_addr, data_size);
	
	gfree(handle, data_addr);
	gfree(handle, k.code_addr);
	gfree(handle, k.cmem_addr);
	gfree(handle, k.lmem_addr);
	
	gclose(handle);
 
	if ((result[0] & 0xffffffff) == 0xdeadcafe)
		return 0;
	else
		return -1;
}
