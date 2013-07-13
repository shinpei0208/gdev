/*
 * Copyright 2013 Yusuke Fujii
 * Ritsumeikan University, Japan
 *
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
#define FCH __gdev_fire_ring(ctx);while(1)

#include "gdev_nvidia_nve4.h"


/* static objects. */
static struct gdev_compute gdev_compute_nve4;

struct gdev_nve4_query {
    uint32_t sequence;
    uint32_t pad;
    uint64_t timestamp;
};

void nve4_compute_setup(struct gdev_device *gdev)
{
    gdev->compute = &gdev_compute_nve4;
}

/**
 * Align a value, only works pot alignemnts.
 */
    static inline int
align(int value, int alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}  


#ifdef GDEV_DEBUG
#define u64 long long unsigned int /* to avoid warnings in user-space */
static void __nve4_launch_debug_print(struct gdev_kernel *kernel)
{
    int i;
    GDEV_PRINT("code_addr = 0x%llx\n", (u64) kernel->code_addr);
    GDEV_PRINT("code_size = 0x%llx\n", (u64) kernel->code_size);
    GDEV_PRINT("code_pc = 0x%x\n", kernel->code_pc);
    for (i = 0; i < kernel->cmem_count; i++) {
	GDEV_PRINT("cmem[%d].addr = 0x%llx\n", i, (u64) kernel->cmem[i].addr);
	GDEV_PRINT("cmem[%d].size = 0x%x\n", i, kernel->cmem[i].size);
	GDEV_PRINT("cmem[%d].offset = 0x%x\n", i, kernel->cmem[i].offset);
    }
    GDEV_PRINT("param_size = 0x%x\n", kernel->param_size);
    for (i = 0; i < kernel->param_size/4; i++)
	GDEV_PRINT("param_buf[%d] = 0x%x\n", i, kernel->param_buf[i]);
    GDEV_PRINT("lmem_addr = 0x%llx\n", (u64) kernel->lmem_addr);
    GDEV_PRINT("lmem_size_total = 0x%llx\n", (u64) kernel->lmem_size_total);
    GDEV_PRINT("lmem_size = 0x%x\n", kernel->lmem_size);
    GDEV_PRINT("lmem_size_neg = 0x%x\n", kernel->lmem_size_neg);
    GDEV_PRINT("lmem_base = 0x%x\n", kernel->lmem_base);
    GDEV_PRINT("smem_size = 0x%x\n", kernel->smem_size);
    GDEV_PRINT("smem_base = 0x%x\n", kernel->smem_base);
    GDEV_PRINT("warp_stack_size = 0x%x\n", kernel->warp_stack_size);
    GDEV_PRINT("warp_lmem_size = 0x%x\n", kernel->warp_lmem_size);
    GDEV_PRINT("reg_count = 0x%x\n", kernel->reg_count);
    GDEV_PRINT("bar_count = 0x%x\n", kernel->bar_count);
    GDEV_PRINT("grid_id = 0x%x\n", kernel->grid_id);
    GDEV_PRINT("grid_x = 0x%x\n", kernel->grid_x);
    GDEV_PRINT("grid_y = 0x%x\n", kernel->grid_y);
    GDEV_PRINT("grid_z = 0x%x\n", kernel->grid_z);
    GDEV_PRINT("block_x = 0x%x\n", kernel->block_x);
    GDEV_PRINT("block_y = 0x%x\n", kernel->block_y);
    GDEV_PRINT("block_z = 0x%x\n", kernel->block_z);
}
#endif

#ifdef GDEV_DEBUG
static void __nve4_launch_debug_print_desc(struct gdev_nve4_compute_desc *desc)
{
    const uint32_t *data=(const uint32_t *)desc;
    int i;
    KEP_PRINT("-----Launch-DESCRIPTION------\n");
    for(i=0;i<8;i++){
	KEP_PRINT(" unk[%d] \t\t= 0x%x\n", i, desc->unk0[i]);
    }
    KEP_PRINT(" entry \t\t= 0x%x\n",  desc->entry);
    for(i=0;i<3;i++){
	KEP_PRINT(" unk9[%d] \t\t= 0x%x\n", i, desc->unk9[i]);
    }
    KEP_PRINT(" griddim_x \t= 0x%x\n",  desc->griddim_x);
    KEP_PRINT(" unk12 \t\t= 0x%x\n",  desc->unk12);
    KEP_PRINT(" griddim_y \t= 0x%x\n",  desc->griddim_y);
    KEP_PRINT(" grdddim_z \t= 0x%x\n",  desc->griddim_z);
    for(i=0;i<3;i++){
	KEP_PRINT(" unk14[%d] \t\t= 0x%x\n",i,  desc->unk14[i]);
    }
    KEP_PRINT(" shared_size \t= 0x%x\n",  desc->shared_size);
    KEP_PRINT(" unk15 \t\t= 0x%x\n",  desc->unk15);
    KEP_PRINT(" unk16 \t\t= 0x%x\n",  desc->unk16);
    KEP_PRINT(" blockdim_x \t= 0x%x\n",  desc->blockdim_x);
    KEP_PRINT(" blockdim_y \t= 0x%x\n",  desc->blockdim_y);
    KEP_PRINT(" blockdim_z \t= 0x%x\n",  desc->blockdim_z);
    KEP_PRINT(" cb_mask \t\t= 0x%x\n",  desc->cb_mask);

    KEP_PRINT(" unk20_8 \t\t= 0x%x\n",  desc->unk20_8);
    KEP_PRINT(" cache_split \t= 0x%x\n",  desc->cache_split);

    KEP_PRINT(" unk20_31 \t\t= 0x%x\n",  desc->unk20_31);
    for(i=0;i<8;i++){
	KEP_PRINT(" unk21[%d] \t\t= 0x%x\n", i, desc->unk21[i]);
    }
    for(i=0;i<8;i++){
	KEP_PRINT(" cb[%d].addr_h\t= 0x%x\n", i, desc->cb[i].address_h);
	KEP_PRINT(" cb[%d].addr_l\t= 0x%x\n", i, desc->cb[i].address_l);
	KEP_PRINT(" cb[%d].size\t= 0x%x\n", i, desc->cb[i].size);
    }
    KEP_PRINT(" local_size_p \t= 0x%x\n",  desc->local_size_p);
    KEP_PRINT(" unk45_20 \t\t= 0x%x\n",  desc->unk45_20);
    KEP_PRINT(" bar_alloc \t= 0x%x\n",  desc->bar_alloc);
    KEP_PRINT(" local_size_n \t= 0x%x\n",  desc->local_size_n);

    KEP_PRINT(" unk46_20 \t\t= 0x%x\n",  desc->unk46_20);
    KEP_PRINT(" gpr_alloc \t= 0x%x\n",  desc->gpr_alloc);

    KEP_PRINT(" cstack_size \t= 0x%x\n",  desc->cstack_size);
    KEP_PRINT(" unk47_20 \t\t= 0x%x\n",  desc->unk47_20);
    for(i=0;i<16;i++){
	KEP_PRINT(" unk48[%02d] \t= 0x%x\n",i,  desc->unk47_20);
    }
    KEP_PRINT("------------------------------\n");
    KEP_PRINT("-----DESC_PRINT such as MESA\n");

    for (i = 0; i < sizeof(*desc); i += 4) {
	if (data[i / 4]) {
	    KEP_PRINT("[%x]: 0x%08x\n", i, data[i / 4]);
	}

    }
}
#endif

static struct gdev_nve4_compute_desc* nve4_launch_desc_setup(struct gdev_ctx *ctx, struct gdev_kernel *k){

    struct gdev_nve4_compute_desc *desc;
    struct gdev_vas *vas = ctx->vas;
    struct gdev_device *gdev = vas->gdev;

    uint32_t cache_split;
    uint32_t mp_limit;
    int x;

    int ad;
    /* setup cache_split so that it'll allow 3 blocks (16 warps each) per 
       SM for maximum occupancy. */
    cache_split = k->smem_size > 16 * 1024 ? 3 : 1;


//    KEP_PRINT("desc_size=%ldbits\n",sizeof(struct gdev_nve4_compute_desc));
//   KEP_PRINT("desc_size=%ldbits\n",sizeof(*desc));
//   KEP_PRINT("desc_gpuaddr=0x%08x\n",ctx->desc.addr);
    desc = (struct gdev_nve4_compute_desc *)ctx->desc.map;
    memset(desc, 0, sizeof(*desc));

    desc->unk0[7]	= 0xbc000000; //default
    desc->entry		= k->code_pc;//entry is offset from CODE_ADDRES  k->code_addr>>8;

    //desc->entry	=0xfeaaa00;//	= k->code_pc;
    desc->unk9[2]	= 0x44014000; //default
    desc->griddim_x     = k->grid_x; 
    // desc->unk12=0;//0x30[31:30]
    desc->griddim_y 	= k->grid_y;
    desc->griddim_z 	= k->grid_z;
    //    desc->unk14[0]=
    //    desc->unk14[1]=
    //    desc->unk14[2]=
    desc->shared_size 	= align(k->smem_size, 0x100);
    //    desc->unk15=
    //    desc->unk16=
    desc->blockdim_x	= k->block_x;
    desc->blockdim_y	= k->block_y;
    desc->blockdim_z	= k->block_z;
    desc->unk20_8=0;//
    desc->cache_split 	= cache_split;
    desc->unk20_31=0;//
    //desc->cb[]=;
    desc->local_size_p	= align(k->lmem_size, 0x10);
    desc->unk45_20=0;
    desc->bar_alloc 	= k->bar_count;
    desc->local_size_n	= k->lmem_size_neg;
    desc->unk46_20=0;
    desc->gpr_alloc	= k->reg_count;
    desc->cstack_size	= k->warp_stack_size+0x600;//gallium 0x800 
    desc->unk47_20	= 0x300;//default
    //desc->unk48[0]=
    //parm=compute object


    desc->cb_mask=0;//mask init
    for (x = 0; x < k->cmem_count && x < 8; x++){
	if(!k->cmem[x].addr || !k->cmem[x].size)
	    continue;

	desc->cb[x].size = k->cmem[x].size;
	desc->cb[x].address_h = k->cmem[x].addr >> 32;
	desc->cb[x].address_l = k->cmem[x].addr;
	desc->cb_mask |= (1 << x);
    }
   // printf("%08x\n",desc);
   // printf("param_%08x\n",k->param_size/4);

    int i;
    if(k->param_size){
	__gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x188, 2);
	__gdev_out_ring(ctx,k->cmem[0].addr >>32); //UPLOAD_DST_ADDRESS_HIGH //fix value
	__gdev_out_ring(ctx,k->cmem[0].addr ); //UPLOAD_DST_ADDRESS_LOW
	__gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x180, 2);
	__gdev_out_ring(ctx, k->param_size); //UPLOAD_LINE_LENGTH_IN
	__gdev_out_ring(ctx, 1); //UPLOAD_LINE_COUNT

	/* fix this
	k->param_buf[0] = k->smem_base;
	k->param_buf[1] = k->lmem_base;
	k->param_buf[2] = k->block_x;
	k->param_buf[3] = k->block_y;
	k->param_buf[4] = k->block_z;
	k->param_buf[5] = k->grid_x;
	k->param_buf[6] = k->grid_y;
	k->param_buf[7] = k->grid_z;
	*/

	k->param_buf[0xa] = k->block_x;
	k->param_buf[0xb] = k->block_y;
	k->param_buf[0xc] = k->block_z;
	k->param_buf[0xd] = k->grid_x;
	k->param_buf[0xe] = k->grid_y;
	k->param_buf[0xf] = k->grid_z;

    }
    __gdev_begin_ring_nve4_1l(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1b0,1 + (k->param_size/4));
    __gdev_out_ring(ctx, (0x20<<1) | 1); // EXEC(EXEC_LINEAR)
    for(i=0;i < k->param_size/4 ;i++){
	__gdev_out_ring(ctx, k->param_buf[i]);
//	printf("param[0x%03x]:0x%08x\n",i,k->param_buf[i]);
    }

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
    __gdev_out_ring(ctx, 0x1000);
    __gdev_fire_ring(ctx);

#if 1
    /* nve4 constant buffer param upload */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x188, 2);
    __gdev_out_ring(ctx,k->cmem[1].addr >>32); //UPLOAD_DST_ADDRESS_HIGH //fix value
    __gdev_out_ring(ctx,k->cmem[1].addr); //UPLOAD_DST_ADDRESS_LOW
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x180, 2);
    __gdev_out_ring(ctx, 7*4); //UPLOAD_LINE_LENGTH_IN
    __gdev_out_ring(ctx, 1); //UPLOAD_LINE_COUNT

    __gdev_begin_ring_nve4_1l(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1b0,8);
    __gdev_out_ring(ctx, (0x20<<1) | 1); // EXEC(EXEC_LINEAR)
    __gdev_out_ring(ctx, desc->blockdim_x);
    __gdev_out_ring(ctx, desc->blockdim_y);
    __gdev_out_ring(ctx, desc->blockdim_z);
    __gdev_out_ring(ctx, desc->griddim_x);
    __gdev_out_ring(ctx, desc->griddim_y);
    __gdev_out_ring(ctx, desc->griddim_z);
    __gdev_out_ring(ctx,0);
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
    __gdev_out_ring(ctx, 0x1000);//1);// | 1); 0x1000 = FLUSH_CB, 0x1 = FLUSH_CODE
    __gdev_fire_ring(ctx);
#endif

    return desc;
}


static int nve4_launch(struct gdev_ctx *ctx, struct gdev_kernel *k)
{


    struct gdev_nve4_compute_desc *desc;

    struct gdev_vas *vas = ctx->vas;
    struct gdev_device *gdev = vas->gdev;

    uint32_t mp_limit;
    int x;
    int ret;


    /* compute desc setup */
    desc = nve4_launch_desc_setup(ctx, k);

#ifdef GDEV_DEBUG
    __nve4_launch_debug_print_desc(desc);
    __nve4_launch_debug_print(k);
#endif

    /* hardware limit. get */
    gdev_query(gdev, GDEV_NVIDIA_QUERY_MP_COUNT, &mp_limit);

    /* local (temp) memory setup */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x790, 2);
    __gdev_out_ring(ctx, k->lmem_addr >> 32); /* TEMP_ADDRESS_HIGH*/
    __gdev_out_ring(ctx, k->lmem_addr); /* TEMP_ADDRESS_LOW */

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2e4, 2);
    __gdev_out_ring(ctx, ( k->lmem_size_total/ mp_limit)>>32); /* MP_TEMP_SIZE_HIGH */
    __gdev_out_ring(ctx, ( k->lmem_size_total/ mp_limit)); /* MP_TEMP_SIZE_LOW*/

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2f0, 2);
    __gdev_out_ring(ctx, ( k->lmem_size_total/ mp_limit)>>32); /* MP_TEMP_SIZE_HIGH */
    __gdev_out_ring(ctx, ( k->lmem_size_total/ mp_limit)); /* MP_TEMP_SIZE_LOW*/

    /* local memory base */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x77c, 1);
    __gdev_out_ring(ctx, k->lmem_base); /* LOCAL_BASE */

    /* shared memory setup */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x214, 1);
    __gdev_out_ring(ctx, k->smem_base); /* SHARED_BASE */

    /**/
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1608, 2);
    __gdev_out_ring(ctx, k->code_addr >> 32); /* CODE_ADDRESS_HIGH */
    __gdev_out_ring(ctx, k->code_addr); /* CODE_ADDRESS_LOW */

    printf("code_addr:0x%x\n",k->code_addr);
    printf("code_addr:0x%x\n",k->code_addr);

#define NVE4_CP_INPUT_MS_OFFSETS 0x10c0
#if 0
    /* MS sample coordinate offsets: these do not work with _ALT modes ! */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE,0x188, 2);
    __gdev_out_ring (ctx, k->cmem[0].addr>>32);
    __gdev_out_ring (ctx, k->cmem[0].addr + NVE4_CP_INPUT_MS_OFFSETS);
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE,0x180, 2);
    __gdev_out_ring (ctx, 64);
    __gdev_out_ring (ctx, 1);
    __gdev_begin_ring_nve4_1l(ctx, GDEV_SUBCH_NV_COMPUTE,0x1b0, 17);
    __gdev_out_ring (ctx,1  | (0x20 << 1));
    __gdev_out_ring (ctx, 0); /* 0 */
    __gdev_out_ring (ctx, 0);
    __gdev_out_ring (ctx, 1); /* 1 */
    __gdev_out_ring (ctx, 0);
    __gdev_out_ring (ctx, 0); /* 2 */
    __gdev_out_ring (ctx, 1);
    __gdev_out_ring (ctx, 1); /* 3 */
    __gdev_out_ring (ctx, 1);
    __gdev_out_ring (ctx, 2); /* 4 */
    __gdev_out_ring (ctx, 0);
    __gdev_out_ring (ctx, 3); /* 5 */
    __gdev_out_ring (ctx, 0);
    __gdev_out_ring (ctx, 2); /* 6 */
    __gdev_out_ring (ctx, 1);
    __gdev_out_ring (ctx, 3); /* 7 */
    __gdev_out_ring (ctx, 1);
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1698, 1);
    __gdev_out_ring(ctx, 0x1000 | 1);
    
    
    __gdev_fire_ring(ctx);
#endif



    /* Launch Kernel Code  */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2b4,1);
    __gdev_out_ring(ctx, (int)(ctx->desc.addr)>>8);//DESC_ADDR

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2bc,1);
    __gdev_out_ring(ctx, 0x3);//LAUNCH

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x110, 1);
    __gdev_out_ring(ctx, 0); //GRAPH_SERIALIZE

    __gdev_fire_ring(ctx);



    return 0;
}

static uint32_t nve4_fence_read(struct gdev_ctx *ctx, uint32_t sequence)
{
    return ((struct gdev_nve4_query*)(ctx->fence.map))[sequence].sequence;
}

static void nve4_fence_write(struct gdev_ctx *ctx, int subch, uint32_t sequence)
{
    uint32_t offset = sequence * sizeof(struct gdev_nve4_query);
    uint64_t vm_addr = ctx->fence.addr + offset;
    int intr = 0; /* intr = 1 will cause an interrupt too. */
    switch (subch) {
	case GDEV_SUBCH_NV_COMPUTE:
	    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x110, 1);
	    __gdev_out_ring(ctx, 0); /* SERIALIZE */
	    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x1b00, 4);
	    __gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
	    __gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
	    __gdev_out_ring(ctx, sequence); /* QUERY_SEQUENCE */
	    __gdev_out_ring(ctx, intr << 20); /* QUERY_GET */
	    break;
	case GDEV_SUBCH_NV_P2MF:
	    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_P2MF, 0x1dc, 3);//32c=>1dc
	    __gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
	    __gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
	    __gdev_out_ring(ctx, sequence); /* QUERY_SEQUENCE */
	    break;
	case GDEV_SUBCH_NV_PCOPY0:
	    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1, 0x240, 3);
	    __gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
	    __gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
	    __gdev_out_ring(ctx, sequence); /* QUERY_COUNTER */
	    break;
#ifdef GDEV_NVIDIA_USE_PCOPY1
	case GDEV_SUBCH_NV_PCOPY1:
	    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1, 0x338, 3);
	    __gdev_out_ring(ctx, vm_addr >> 32); /* QUERY_ADDRESS HIGH */
	    __gdev_out_ring(ctx, vm_addr); /* QUERY_ADDRESS LOW */
	    __gdev_out_ring(ctx, sequence); /* QUERY_COUNTER */
	    break;
#endif
    }
    __gdev_fire_ring(ctx);
}

static void nve4_fence_reset(struct gdev_ctx *ctx, uint32_t sequence)
{
    ((struct gdev_nve4_query*)(ctx->fence.map))[sequence].sequence = ~0;
}

unsigned min2(unsigned a,unsigned b){
    return a>b? b:a;
}

static void nve4_memcpy_p2mf(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size){
#if 1
    KEP_PRINT("not implemented\n");
#else
    //this function is upload only
    unsigned nr=16;
    unsigned count = (size + 3) /4;	
    int i;
    unsigned offset;
    // while(count){

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_P2MF, 0x188, 2);
    __gdev_out_ring(ctx, (dst_addr+offset) >>32);//P2MF_DST_ADDRESS_HIGH
    __gdev_out_ring(ctx, dst_addr+offset);//P2MF_DST_ADDRESS_LOW

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_P2MF, 0x180, 2);
    __gdev_out_ring(ctx, min2(size,nr*4)); //LINE_LENGTH_IN
    __gdev_out_ring(ctx, 1);
    /* must not be interrupted (trap on QUERY fence, 0x50 works however) */

    __gdev_begin_ring_nve4_1l(ctx, GDEV_SUBCH_NV_P2MF, 0x180, 1);
    __gdev_out_ring(ctx, 0x1001);

    /*for(i=0;i<nr;i++){
      }*/

    count-=nr;
    src_addr+=nr;
    offset+=nr*4;
    size-=nr;
    // }
#endif

    __gdev_fire_ring(ctx);

}


static void _nve4_memcpy_m2mf_transfer_rect(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size){

    uint32_t exec;
    uint32_t tile_mode=0; //what's this?
    uint32_t page_size = 0x1000;
    uint32_t page_count = size/page_size;
    uint32_t line_count = (page_count > 2047) ? 2047:page_count;
    uint32_t pitch = 0x80000;
    uint32_t ycnt = size / pitch;
    uint32_t rem_size = size - pitch * ycnt;;

#if 0
    KEP_PRINT("not implemented\n");
#else
    exec = 0x200 /* 2D_ENABLED */ | 0x6 /* UNK */;

    if(page_count == line_count && rem_size == 0)
	exec |= 0x100; /* DST_MODE_2D_LINEAR  */
    else
	exec |= 0x080; /*SRC_MODE_2D_LINEAR*/

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1,0x070c,6);
    __gdev_out_ring(ctx, 0x1000 | tile_mode );// | DST_tile_mode
    __gdev_out_ring(ctx, pitch);//DST_pitch
    __gdev_out_ring(ctx, page_size);//DST_HEIGHT
    __gdev_out_ring(ctx, ycnt);//DST_DEPTH
    __gdev_out_ring(ctx, pitch);//DST_Z
    __gdev_out_ring(ctx, page_size);//DST_Y << 16, DST_X *CPP

    __gdev_begin_ring_nve4(ctx,GDEV_SUBCH_NV_PCOPY1,0x0728,6);
    __gdev_out_ring(ctx, 0x1000 | tile_mode );// SRC_TILE_MODE
    __gdev_out_ring(ctx, pitch);//SRC_pitch
    __gdev_out_ring(ctx, page_size);//SRC_HEIGHT
    __gdev_out_ring(ctx, ycnt);//SRC_DEPTH
    __gdev_out_ring(ctx, 1);//SRC_Z
    __gdev_out_ring(ctx, page_size);//SRC_Y << 16, SRC_X *CPP

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1,0x400, 8);
    __gdev_out_ring(ctx, src_addr >> 32); // SRC_BASE_HIGH
    __gdev_out_ring(ctx, src_addr); //SRC_BASE_LOW
    __gdev_out_ring(ctx, 401+(dst_addr >> 32)); // DST_BASE_HIGH
    __gdev_out_ring(ctx, dst_addr); //DST_BASE_LOW
    __gdev_out_ring(ctx, pitch); //SRC_PITCH
    __gdev_out_ring(ctx, pitch); //DST_PITCH
    __gdev_out_ring(ctx, pitch);//nblocksx * cpp
    __gdev_out_ring(ctx, ycnt);//nblocksy

    __gdev_begin_ring_nve4(ctx,4,0x300,1);
    __gdev_out_ring(ctx,exec);

    __gdev_fire_ring(ctx);
#endif

    return;
}

static void nve4_copy_linear(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size){

    uint32_t page_size = 0x1000;
    uint32_t page_count = size / page_size;
    uint32_t rem_size = size - page_size * page_count;
    
    int line_count = (page_count > 2047-1) ? 2047-1 : page_count;
   
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1, 0x400,4);
    __gdev_out_ring(ctx, src_addr>>32);
    __gdev_out_ring(ctx, src_addr);
    __gdev_out_ring(ctx, dst_addr>>32);
    __gdev_out_ring(ctx, dst_addr);
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1, 0x418,1);
    __gdev_out_ring(ctx, size);
    __gdev_begin_ring_nve4_1l(ctx, GDEV_SUBCH_NV_PCOPY1, 0x300,1);
    __gdev_out_ring(ctx, 0x186);/*  */
    __gdev_fire_ring(ctx);
}

static void nve4_memcpy_m2mf(struct gdev_ctx *ctx, uint64_t dst_addr, uint64_t src_addr, uint32_t size)
{
    return nve4_copy_linear(ctx, dst_addr, src_addr,size);//need referctoring
}

static void nve4_membar(struct gdev_ctx *ctx)
{
    /* this must be a constant method. */
    __gdev_begin_ring_nve4_const(ctx, GDEV_SUBCH_NV_COMPUTE, 0x21c, 2);
    __gdev_out_ring(ctx, 4); /* MEM_BARRIER */
    // what is this ? #1
    __gdev_out_ring(ctx, 0x1011); /* maybe wait for everything? */
    __gdev_fire_ring(ctx);
}

static void nve4_notify_intr(struct gdev_ctx *ctx)
{
    uint64_t addr = ctx->notify.addr;

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x110, 1);
    __gdev_out_ring(ctx, 0); /* SERIALIZE */
    // PGRAPH object=> 104 is notify
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x104, 3);
    __gdev_out_ring(ctx, addr >> 32); /* NOTIFY_HIGH_ADDRESS */
    __gdev_out_ring(ctx, addr); /* NOTIFY_LOW_ADDRESS */
    __gdev_out_ring(ctx, 1); /* WRITTEN_AND_AWAKEN */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x100, 1);
    __gdev_out_ring(ctx, ctx->cid); /* NOP */

    __gdev_fire_ring(ctx);
}

static void nve4_init(struct gdev_ctx *ctx)
{
    int i;
    uint64_t mp_limit;
    struct gdev_vas *vas = ctx->vas;
    struct gdev_device *gdev = vas->gdev;

    /* initialize the fence values. */
    for (i = 0; i < GDEV_FENCE_COUNT; i++)
	nve4_fence_reset(ctx, i);

    /* clean the FIFO. */
    for (i = 0; i < 128/4; i++)
	__gdev_out_ring(ctx, 0);
    __gdev_fire_ring(ctx);

    /* setup subchannels. */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_P2MF, 0, 1);
    __gdev_out_ring(ctx, 0xa040); /* P2MF of GK110 */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0, 1);
    __gdev_out_ring(ctx, 0xa0c0); /* COMPUTE */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1, 0, 1);
    __gdev_out_ring(ctx, 0xa0b5); /* PCOPY1 */

    /* enable PCOPY only when we are in the kernel atm... */
#if 0
#ifdef __KERNEL__
#if LINUX_VERSION_CODE < KERNEL_VERSION(3,7,0)
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY0, 0, 1);
    __gdev_out_ring(ctx, 0x490b5); /* PCOPY0 */
#ifdef GDEV_NVIDIA_USE_PCOPY1
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_PCOPY1, 0, 1);
    __gdev_out_ring(ctx, 0x590b8); /* PCOPY1 */
#endif
#endif
#endif
#endif

    __gdev_fire_ring(ctx);
    /* the blob places NOP at the beginning. */
    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x110, 1);
    __gdev_out_ring(ctx, 0); /* GRAPH_NOP */
    /* hardware limit. get */
    gdev_query(gdev, GDEV_NVIDIA_QUERY_MP_COUNT, &mp_limit);

    __gdev_fire_ring(ctx);

    __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x310, 1);//UNK0310
    __gdev_out_ring(ctx, 0x300);// obj_class >= NVF0_COMPUTE_CLASS? 0x400:0x300

    /*texture is not supported*/
    //   __gdev_begin_ring_nve4(ctx, GDEV_SUBCH_NV_COMPUTE, 0x2608,1);//TEX_CB_INDEX
    //  __gdev_out_ring(ctx, 0);

    __gdev_fire_ring(ctx);
}

static struct gdev_compute gdev_compute_nve4 = {
    .launch = nve4_launch,
    .fence_read = nve4_fence_read,
    .fence_write = nve4_fence_write,
    .fence_reset = nve4_fence_reset,
    .memcpy = nve4_memcpy_m2mf,
    .memcpy_async = nve4_memcpy_p2mf,
    .membar = nve4_membar,
    .notify_intr = nve4_notify_intr,
    .init = nve4_init,
};
