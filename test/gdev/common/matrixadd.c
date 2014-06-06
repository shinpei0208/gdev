#ifdef __KERNEL__ /* just for measurement */
#include <linux/vmalloc.h>
#include <linux/time.h>
#define printf printk
#define malloc vmalloc
#define free vfree
#define gettimeofday(x, y) do_gettimeofday(x)
#else /* just for measurement */
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#endif
#include "gdev_api.h"

/*
00000000: 00005de4 28004404     mov b32 $r1 c1[0x100]
00000008: 98009c04 2c000000     mov b32 $r2 $ctaidy
00000010: 88011c04 2c000000     mov b32 $r4 $tidy
00000018: 94001c04 2c000000     mov b32 $r0 $ctaidx
00000020: 8400dc04 2c000000     mov b32 $r3 $tidx
00000028: 30209c03 20084000     add $r2 (mul u32 $r2 u32 c0[0xc]) $r4
00000030: 20001c03 20064000     add $r0 (mul u32 $r0 u32 c0[0x8]) $r3
00000038: e021dc03 188e4000     set $p0 0x1 lt u32 $r2 c0[0x38]
00000040: e001dc03 18804000     set $p0 0x1 lt u32 $r0 c0[0x38] and $p0
00000048: 000021e7 80000000     (not $p0) exit
00000050: e000dc03 20044000     add $r3 (mul u32 $r0 u32 c0[0x38]) $r2
00000058: 10015de2 18000000     mov b32 $r5 0x4
00000060: 10311ce3 5000c000     mul high $r4 s32 $r3 s32 0x4
00000068: 80319ca3 200b8000     add $r6 $c (mul s32 $r3 s32 $r5) c0[0x20]
00000070: 9041dc43 48004000     add b32 $r7 $r4 c0[0x24] $c
00000078: a0321ca3 200b8000     add $r8 $c (mul s32 $r3 s32 $r5) c0[0x28]
00000080: 00609c85 84000000     ld b32 $r2 ca g[$r6d]
00000088: b0425c43 48004000     add b32 $r9 $r4 c0[0x2c] $c
00000090: c0329ca3 200b8000     add $r10 $c (mul s32 $r3 s32 $r5) c0[0x30]
00000098: 00801c85 84000000     ld b32 $r0 ca g[$r8d]
000000a0: d042dc43 48004000     add b32 $r11 $r4 c0[0x34] $c
000000a8: 00201c03 48000000     add b32 $r0 $r2 $r0
000000b0: 00a01c85 94000000     st b32 wb g[$r10d] $r0
000000b8: 00001de7 80000000     exit
 */
uint32_t kcode[] = {
	0x00005de4,
	0x28004404,
	0x98009c04,
	0x2c000000,
	0x88011c04,
	0x2c000000,
	0x94001c04,
	0x2c000000,
	0x8400dc04,
	0x2c000000,
	0x30209c03,
	0x20084000,
	0x20001c03,
	0x20064000,
	0xe021dc03,
	0x188e4000,
	0xe001dc03,
	0x18804000,
	0x000021e7,
	0x80000000,
	0xe000dc03,
	0x20044000,
	0x10015de2,
	0x18000000,
	0x10311ce3,
	0x5000c000,
	0x80319ca3,
	0x200b8000,
	0x9041dc43,
	0x48004000,
	0xa0321ca3,
	0x200b8000,
	0x00609c85,
	0x84000000,
	0xb0425c43,
	0x48004000,
	0xc0329ca3,
	0x200b8000,
	0x00801c85,
	0x84000000,
	0xd042dc43,
	0x48004000,
	0x00201c03,
	0x48000000,
	0x00a01c85,
	0x94000000,
	0x00001de7,
	0x80000000
};

uint32_t c0[] = {
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0,
	0x0
};

#define PARAM_SIZE 0x3c
#define STACK_DEPTH 0xc
#define LOCAL_SIZE 0x0
#define SHARED_SIZE 0x0
#define REG_COUNT 0xc
#define BARRIER_COUNT 0x0
#define NVCC_PARAM_OFFSET 0x20

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

int gdev_test_matrixadd(uint32_t *a, uint32_t *b, uint32_t *c, int n)
{
	int i, j, idx;
	uint32_t id;
	uint32_t mp_count = 0;
	uint32_t code_size, a_size, b_size, c_size;
	uint32_t param_buf[PARAM_SIZE];
	uint64_t a_addr, b_addr, c_addr;
	uint64_t result[3];

	Ghandle handle;
	struct gdev_kernel k;

	/* initialize A[] & B[] */
	for (i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			idx = i * n + j;
			a[idx] = i;
			b[idx] = j;
		}
	}

	if (!(handle = gopen(0))) {
		return -1;
	}
	
	a_size = n * n * sizeof(uint32_t);
	b_size = n * n * sizeof(uint32_t);
	c_size = n * n * sizeof(uint32_t);

	if (!(a_addr = gmalloc(handle, a_size)))
		return -1;
	if (!(b_addr = gmalloc(handle, b_size)))
		return -1;
	if (!(c_addr = gmalloc(handle, c_size)))
		return -1;

	code_size = sizeof(kcode);
	if (code_size & 0xff)
		k.code_size = (code_size + 0x100) & ~0xff;
	if (!(k.code_addr = gmalloc(handle, k.code_size)))
		return -1;
	k.code_pc = 0;

	k.cmem[0].size = PARAM_SIZE;
	if (k.cmem[0].size == 0 || k.cmem[0].size & 0xff)
		k.cmem[0].size = (k.cmem[0].size + 0x100) & ~0xff;
	if (!(k.cmem[0].addr = gmalloc(handle, k.cmem[0].size)))
		return -1;
	k.cmem[0].offset = 0;
	for (i = 1; i < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		k.cmem[i].addr = 0;
		k.cmem[i].size = 0;
		k.cmem[i].offset = 0;
	}
	k.cmem_count = GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT;
	k.param_size = PARAM_SIZE;
	k.param_buf = c0;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 0] = a_addr;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 1] = a_addr >> 32;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 2] = b_addr;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 3] = b_addr >> 32;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 4] = c_addr;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 5] = c_addr >> 32;
	k.param_buf[NVCC_PARAM_OFFSET/4 + 6] = n;

	k.lmem_size = LOCAL_SIZE;
	if (k.lmem_size & 0xf)
		k.lmem_size = (k.lmem_size + 0x10) & ~0xf;
	k.lmem_size_neg = 0; /* just random */
	if (k.lmem_size_neg & 0xf)
		k.lmem_size_neg = (k.lmem_size_neg + 0x10) & ~0xf;
	k.lmem_base = 0x01000000;
	k.smem_size = SHARED_SIZE;
	if (k.smem_size & 0x7f)
		k.smem_size = (k.smem_size + 0x80) & (~0x7f);
	k.smem_base = 0x0;
	
	k.warp_stack_size = (STACK_DEPTH + 0x1000) & (~0xfff);
	
	/* FIXME: per-thread warp size may differ from 32. */
	k.warp_lmem_size = 32 * (k.lmem_size + k.lmem_size_neg) + k.warp_stack_size; 
	
	/* FIXME: the number of active warps may differ from 48. */
	gquery(handle, GDEV_NVIDIA_QUERY_MP_COUNT, (uint64_t *)&mp_count);
	if (!mp_count) mp_count = 32;
	k.lmem_size_total = 48 * mp_count * k.warp_lmem_size;
	k.lmem_size_total = __round_up_pow2(k.lmem_size_total);
	if (!(k.lmem_addr = gmalloc(handle, k.lmem_size_total)))
		return -1;

	k.reg_count = REG_COUNT;
	k.bar_count = BARRIER_COUNT;
	k.grid_id = 1;
	
	k.block_x = n < 16 ? n : 16;
	k.block_y = n < 16 ? n : 16;
	k.block_z = 1;
	k.grid_x = n / k.block_x;
	if (n % k.block_x != 0)
		k.grid_x++;
	k.grid_y = n / k.block_y;
	if (n % k.block_y != 0)
		k.grid_y++;
	k.grid_z = 1;
	
	gmemcpy_to_device(handle, k.code_addr, kcode, k.code_size);
	gmemcpy_to_device(handle, a_addr, a, a_size);
	gmemcpy_to_device(handle, b_addr, b, b_size);
	
	glaunch(handle, &k, &id);
	gsync(handle, id, NULL);
	
	gmemcpy_from_device(handle, c, c_addr, c_size);

	gfree(handle, a_addr);
	gfree(handle, b_addr);
	gfree(handle, c_addr);
	gfree(handle, k.code_addr);
	gfree(handle, k.cmem[0].addr);
	gfree(handle, k.lmem_addr);
	
	gclose(handle);

	i = j = idx = 0;
	while (i < n) {
		while (j < n) {
			idx = i * n + j;
			if (c[idx] != a[idx] + b[idx]) {
				printf("c[%d] = %d\n", idx, c[idx]);
				printf("a[%d]+b[%d] = %d\n", idx, idx, a[idx]+b[idx]);
				return -1;
			}
			j++;
		}
		i++;
	}

	return 0;
}
