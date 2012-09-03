
#include "asminsn.h"
#include "mmio.h"
#include "regs_gpc.h"
#include "types.h"


// 0..18 stamp
int stamp[] = {
    0x2072614d,
    0x32203120,
    0x00303130,
    0x323a3031,
    0x39333a39,
    0x00000000
};

// 18
int unk18 = 0x186a0;
// 1c
int testres = 0x1000;
// 20..13c mmio tables
int mmio_table1[] = {
    0x00000380, // #0
    0x14000400, // #1
    0x20000450, // #2
    0x00000600, // #3
    0x00000684, // #4
    0x10000700, // #5
    0x00000800, // #6
    0x08000808, // #7
    0x00000828, // #8
    0x00000830, // #9
    0x000008d8, // #10
    0x000008e0, // #11
    0x140008e8, // #12
    0x0000091c, // #13
    0x08000924, // #14
    0x00000b00, // #15
    0x14000b08, // #16
    0x00000bb8, // #17
    0x00000c08, // #18
    0x1c000c10, // #19
    0x00000c80, // #20
    0x00000c8c, // #21
    0x08001000, // #22
    0x00001014  // #23
};
int nr_mmio_table1 = 0x18; // 24

int mmio_table2[] = {
    0x00000048, // #0
    0x00000064, // #1
    0x00000088, // #2
    0x14000200, // #3
    0x0400021c, // #4
    0x14000300, // #5
    0x000003d0, // #6
    0x040003e0, // #7
    0x08000400, // #8
    0x00000420, // #9
    0x000004e8, // #10
    0x000004f4, // #11
    0x04000520, // #12
    0x0c000604, // #13
    0x4c000644, // #14
    0x00000698, // #15
    0x04000750  // #16
};
int nr_mmio_table2 = 0x11; // 17

int mmio_table3[] = {
    0x18000a00, // #0
    0x18000a20, // #1
    0x18000a40, // #2
    0x18000a60, // #3
    0x18000a80, // #4
    0x18000aa0, // #5
    0x18000ac0, // #6
    0x18000ae0  // #7
};
int nr_mmio_table3 = 0x8;

int mmio_table4[] = {
    0x00000384, // #0
    0x000004a0, // #1
    0x00000604, // #2
    0x00000680, // #3
    0x00000714, // #4
    0x0000081c, // #5
    0x000008c8, // #6
    0x00000b04  // #7
};
int nr_mmio_table4 = 0x8;

int mmio_table5[] = {
    0x00000054, // #0
    0x0000040c, // #1
    0x000004a8, // #2
    0x00000600  // #3
};
int nr_mmio_table5 = 0x4;

int mmio_table6[] = {
    0x7c000040,
    0x400000c0
};
int nr_mmio_table6 = 0x2;

int mmio_table7[] = {
    0x14000000
};
int nr_mmio_table7 = 0x1;

struct pair 
{
    int a, b;
};

struct qe {
    uint32_t cmd;
    uint32_t data;
};

// 13c
int done;
// 148
int ctx1rsz;
// 14c
int unkc14c;
//150
int unkc150;
//154
int unkc154;
//158
int unkc158;

// 15c
int curctxbase;
// 170
int curs3base;
// 174
int curs2base;
// 178
int curs3pres;
// 17c
int curs2pres;
// 180
int myindex;
// 184
int mpcount;
// 188
struct qe queue[8];
// 1c8
int qget;
// 1cc
int qput;
// 1d0
struct qe q2[8];
// 210
int q2get;
// 214
int q2put;
//218
int unkc218;
int unkc21c;

struct temp{
    int temp;
} temp __attribute__ ((aligned (0x100))) = {0};

// 300
struct s3 {
    // 300
    int strsz;
    // 304
    int strands;
    // 308 unk
    int unk308;
    // 30c
    int mmbase;
    // 310 unk
    int unk310;
    // 314..3a0 unk
    struct pair unk314[9];
    // 35c..3a4 unk
    int unk35c[12];
    int unk38c;
    int unk390;
    int unk394;
    int unk398;
    int unk39c;
    int unk3a0;
    // 3a4..3e8
    struct pair unk3a4[9];
    // 3ec
    int unk3ec;
    // 3f0... unk
    int unk3f0;
    int unk3f4;
    int unk3f8;
    // 3fc
    int marker;
} s3 __attribute__ ((aligned (0x100))) = {0};
/*
// 400
struct s4 
{
struct pair unk400;
struct pair unk408;    
struct pair unk410;
struct pair unk418[5];
} s4 __attribute__ ((aligned (0x100))) = {0};

// 500
struct s5 {
int unk500;
int unk504;
int tpcount; // 508
int unk510;
int unk514;
int unk518;
int unk51c;
int unk520;
int unk524;
int unk528;
int unk52c;
int unk530;
int unk534;
int unk538;
int unk53c;
int unk540[8];
//564
int unktemp[36];
int unk5f4;
int unk5f8;
int marker // 5fc
} s5 __attribute__ ((aligned (0x100))) = {0};
*/


//44 unkc13
void unkc13 (uint32_t cmd, uint32_t data){
    //always inlined
    curctxbase = data;
}

//4d unkc17
void unkc17 (uint32_t cmd, uint32_t data){
    //always inlined
    curs3base = data;
}

//56 unkc26
void unkc26 (uint32_t cmd,uint32_t data){
    //always inlined
    curs2base = data;
}

//5f unkc19
void unkc19 (uint32_t cmd,uint32_t data){
    //always inlined
    curs3pres = data;
}

//68 unkc28
void unkc28 (uint32_t cmd,uint32_t data){
    //always inlined
    curs2pres = data;
}

// 71 unkc18
void unkc18(uint32_t cmd,uint32_t data) {
    int temp;
    extr(temp,unkc150,8,15);
    s3.unk3a0 = (data & 0xffff) | temp << 0x10;
}

//8e unkc27
void unkc27(uint32_t cmd, uint32_t data){
    int temp;
    extr(temp,unkc158,8,15);
    s3.unk3a4[8].b = data & 0xffff | (temp << 0x10);
}

//ab mmctxsz
int mmctxsz(int x){
    int res;
    switch(x) {
	case 3:
	    res = (mpcount * 0x37 + 0x40) * 4;
	    break;
	case 5:
	    res = 0xe0;
	    break;
	case 6:
	    res = mpcount * 0x10 + 0xfc;
	    break;
	default:
	    res = 0;
	    break;
    }
    return ((res >> 8) + 1) << 8;
}


//never used
void fa(x){
    mmio_write(0x91c, mmio_table1[1]);
    mmio_write_i(0x91c,0x3f,x);
    mmio_write_i(0x928,0x3f,3);
}
#if 0
//113
void funk113(x){
    //never used
    mmio_write_i(0x91c,0x3f,x);
    mmio_write_i(0x928,0x3f,0x4);
}
//12c
void funk12c(x){
    //never used
    if(x)
	mmio_write_i(0x928,0x3f,0xc);
    else
	mmio_write_i(0x928,0x3f,0xd);
}

#endif

//149 set93c
void set93c(int x, int y) {
    mmio_write_i(0x93c,0x3f,x);
    if (y)
	mmio_write_i(0x928,0x3f,0xa);
    else
	mmio_write_i(0x928,0x3f,0xb);
}

//175
void funk175( int x){
    mmio_write(0x614,0x20);
    unkc21c = x;
    while(--x);
    unkc21c = x;
    mmio_write(0x614,0xa20);
}

//1af never used
void funk1af(int x, int y, int z){
}

//1cc never used
void funk1cc(){
}


// 1da

void funk1da(x) {
    if (x)
	mmio_write(0x430, (x & 0x3fffffff) | 0x80000000);
}
// 200 mmwrq

void mmwrq(uint32_t addr,uint32_t  val,uint32_t z) {
    mmio_write(FUC_MMIO_WRVAL, val);
    mmio_write(FUC_MMIO_CTRL, (addr & 0x3fffffc) | (z & 1) | 0xc0000000);
}


// 22d xfer

void xfer(int base, int offset,int sizel2,int local_addr,int subtrg,int trg,int dir,int wait) {
    int tspec;
    mmio_write(FUC_NEWSCRATCH_SET3, 0x200);
    switch (trg) {
	case 2:
	    tspec = 1;
	    break;
	case 1:
	    switch (subtrg) {
		case 0:
		    tspec = 0x80000002;
		    break;
		case 2:
		    tspec = 0x80000004;
		    break;
		case 3:
		    tspec = 0x80000003;
		    break;
		default:
		    tspec = 0x80000000;
		    break;
	    }
	    break;
	default:
	    tspec = 0;
	    break;
    }
    mmio_write(0xa20, tspec);
    set_xdbase(base);
    set_xtargets(0);
    mmio_write(0x10c, 0);
    if (dir == 1) {
	xdld(offset, (sizel2 << 0x10) | local_addr);
    } else {
	xdst(offset, (sizel2 << 0x10) | local_addr);
    }
    if (wait == 2)
	xdwait();
    mmio_write(0xa20, 0);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x200);	
}

// 2e9 store3

void store3(int wait) {
    mmio_write(FUC_NEWSCRATCH_SET3, 2);
    xfer(0, 0x200 + myindex * 0x100, 6, 0x300, 0, 2, 2, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 2);
}

//334 load3
void load3(int wait) {
    mmio_write(FUC_NEWSCRATCH_SET3, 1);
    xfer(0, 0x200 + myindex * 0x100, 6, 0x300, 0, 2, 1, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 1);
}

// 382 unkc14

void unkc14(uint32_t cmd,uint32_t data) {

    uint32_t base, tm;
    int ctxtemp,i;
    extr(ctxtemp,ctx1rsz,8,15);
    mmio_write(FUC_NEWSCRATCH_SET2, 0x100000);
    s3.mmbase = (data & 0x0ffff) | (ctxtemp << 0x10);
    base = (s3.mmbase & 0x0ffff) + (s3.mmbase >> 0x10);

    for (i = 0; i != s3.strands; i++) {
	tm = s3.unk314[i].a + base;
	mmio_write_i(FUC_STRAND_SAVE_SWBASE, i, tm);
	mmio_write_i(FUC_STRAND_LOAD_SWBASE, i, tm);
    }
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x100000);
}

// 40a mmwr

void mmwr(uint32_t addr,uint32_t val,uint32_t z, uint32_t w) {
    mmio_write(FUC_MMIO_WRVAL, val);
    mmio_write(FUC_MMIO_CTRL, (addr & 0x3fffffc) | (z & 1) | 0xc0000000 | !!w << 29);
    while (mmio_read(FUC_MMIO_CTRL) & 0x80000000);
    if (w)
	while (!(mmio_read(FUC_DONE) & 0x80));
}


void funk47c(int x) {
    mmio_write(0x614, 0x270);
    mmwr(0x41a614, 0x820, 0, 1); // XXX mmwr with 1
    unkc218 = x;
    while(--x);
    unkc218 = x;
    mmio_write(0x614, 0x770);
    mmwr(0x41a614, 0xa20, 0, 1); // XXX mmwr with 1
}

// 4e6 unkc6

void unkc6(uint32_t cmd, uint32_t data) {
    funk47c(8);
    mmio_write(FUC_NEWSCRATCH0, 0x100);
}

//4fc unkc25

void unkc25(uint32_t cmd, uint32_t data){
    uint32_t o;
    mmio_write(FUC_NEWSCRATCH_SET2,0x800);
    o = mmctxsz(6);
    unkc158 += o;
    unkc154 += o;
    s3.unk3a4[8].a += unkc154;
    mmio_write(FUC_NEWSCRATCH0, unkc154);
    mmwr(0x409418, 1 << myindex, 0, 0);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x800);
}

//56d error
void error(uint32_t ecode){
    mmwr(0x409818, ecode, 0, 0);
    mmwr(0x409c1c, 1, 0, 0);
}

//597

void err2(uint32_t ecode){
    error(ecode);
}



#if 0
//59d enq2
void enq2(cmd, data){
    //never used
    mmio_write(FUC_NEWSCRATCH_SET2,0x4);
    if((q2put & 7) == (q2get &7) && (q2put & 8) != (q2get & 8 ))
    {
	err2(0x12);
    }else{
	q2[q2put & 7].data = data;
	q2[q2put & 7].cmd = cmd;
	if(++q2put == 0x10)
	    q2put=0;
    }
    mmio_write(FUC_NEWSCRATCH_CLEAR0,0x4);
}



#endif
// 61c enqueue
void enqueue(uint32_t cmd, uint32_t data) {

    mmio_write(FUC_NEWSCRATCH_SET2, 2);
    if ((qput & 7) == (qget & 7) && (qput & 8) != (qget &8)) {
	// 66a
	err2(0x12);
    } else {
	// 674
	queue[qput & 7].data = data;
	queue[qput & 7].cmd = cmd;
	if (++qput == 0x10) {
	    qput = 0;
	}     
    }
    // 688
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 2);
}


// 69b waitdone_12

void waitdone_12() {
    mmio_write(FUC_NEWSCRATCH_SET3, 0x100);
    while ((mmio_read(FUC_DONE)&0x1000));
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x100);
}
// 6cb setxferbase

void setxferbase(x) {
    waitdone_12();
    mmio_write(FUC_MEM_BASE, x);
}


// 7e6 mmctx

void mmctx (int dir, int y, int z, int *ptraddr, int len) {
    mmio_write(FUC_NEWSCRATCH_SET3, 0x10);
    int *ptr;
    int i,mode,free;
    ptr = (int *)ptraddr;
    switch (y) {
	case 3:
	case 5:
	case 6:
	    // 812
	    mmio_write(FUC_MMCTX_BASE, (0x500000 + myindex * 0x8000) & 0x3ffffff);
	    mode = 1;
	    break;
	case 4:
	case 7:
	    // 852
	    mmio_write(FUC_MMCTX_BASE, (0x504000 + myindex * 0x8000) & 0x3ffffff);
	    mmio_write(FUC_MMCTX_MULTI_STRIDE, 0x800);
	    mmio_write(FUC_MMCTX_MULTI_MASK, (1 << mpcount) - 1);
	    mode = 3;
	    break;
	case 0xb:
	    mmio_write(FUC_MMCTX_BASE, (0x180000 + myindex * 0x1000) & 0x3ffffff);
	    mode = 1;
	    break;
	case 0xc:
	    mmio_write(FUC_MMCTX_BASE, (0x1b8000 + myindex * 0x200) & 0x3ffffff);
	    mode = 1;
	    break;
	default:
	    mode = 0;
    }
    // 8f3
    if (z <= 2)
	//	mmio_write(FUC_MMCTX_CTRL, (dir == 2) << 0x10 | 0x21000);
	mmio_write(FUC_MMCTX_CTRL, 1 << 0x10 | 0x21000);
    // 910
    free = 0;

    for (i = 0; i < len; i++) {
	while (free==0) {
	    free = (mmio_read(FUC_MMCTX_CTRL) ) & 0x1f;
	}
	mmio_write(FUC_MMCTX_QUEUE, ptr[i] | mode);
	free--;
    }

    if (z == 1 || z == 3){
	while (free != 0x10)
	{ 
	    free = mmio_read(FUC_MMCTX_CTRL) & 0x1f;
	}
	while (!(mmio_read(FUC_DONE)&0x20));
    }else {
	//mmio_write(FUC_MMCTX_CTRL, (dir == 2) << 0x10 | 0x41000);
	mmio_write(FUC_MMCTX_CTRL, (dir==2) << 0x10 | 0x41000);
	while(mmio_read(FUC_MMCTX_CTRL) & 0x40000);
    }
    // 9b3
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x10);
}

// a11 waitstr
void waitstr() {
    mmio_write(FUC_NEWSCRATCH_SET3, 0x4000);
    while ((mmio_read(FUC_DONE)&4));
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x4000);
}

// a40
void funka40(int x) {
    mmio_write(FUC_NEWSCRATCH_SET3, 0x8000);
    mmio_write_i(0x91c, 0x3f, x);
    mmio_write_i(0x928, 0x3f, 1);
    waitstr();
    mmio_write_i(0x918, 0x3f, 0);
    mmio_write_i(0x928, 0x3f, 5);
    waitstr();
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x8000);
}

//a96
void funka96(int x) {
    mmio_write(FUC_NEWSCRATCH_SET3, 0x2000);
    mmio_write_i(0x91c, 0x3f, x);
    mmio_write_i(0x928, 0x3f, 1);
    waitstr();
    mmio_write_i(0x918, 0x3f, 0xffffffff);
    mmio_write_i(0x928, 0x3f, 2);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x2000);
}

//ae0 ihbodu
void ihbody(void)
{
    uint32_t intr = mmio_read(0x008);
    if ( intr & 4){
	enqueue(mmio_read(0x068),mmio_read(0x064));
	mmio_write(0x074,0x1);
    }
    //b0f
    if (intr & 0xff00){
	if (intr & 0x4000)
	    mmwrq(0x409c1c, 0x10000, 0);
	//b38
	if (intr & 0x2000)
	    mmwrq(0x409c1c, 0x80000, 0);
	//b51
	if (intr & 0x1000)
	    mmwrq(0x41a69c, 1, 9);
    }
    //b67
    mmio_write(0x004,intr);
}


//c90 strctxsz
int strctxsz (int x, int y) {
    int i,total;
    int tm;

    mmio_write(FUC_NEWSCRATCH_SET3, 0x1000);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xc); 
    waitstr();
    set93c(0xf, 0);
    waitstr();
    set93c(y, 1);
    waitstr();
    funka96(0);
    waitstr();
    set93c(0xf, 0);
    waitstr();
    set93c(3, 1);
    waitstr();
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xd);
    waitstr();

    total = 0;
    for (i = 0; i != 8; i++) {
	// d12
	tm = mmio_read_i(FUC_STRAND_SIZE, i);
	if (tm > 0) {
	    tm <<= 2;
	    if (y == 8) {
		// d41
		s3.unk3a4[i].a = x;
		s3.unk3a4[i].b = tm;
	    } else {
		// d50
		s3.unk314[i].a = x;
		s3.unk314[i].b = tm;
	    }
	    // d5d
	    tm = (tm >> 8) + 1;
	    x += tm;
	    total += tm << 8;
	}
    }
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x1000);
    return total;
}

// da7 unkc16

void unkc16(uint32_t cmd,uint32_t data) {
    int o1,o2;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x400);
    o1 = mmctxsz(5);
    unkc14c += o1;
    unkc150 += o1;
    s3.unk394 = mmio_read(FUC_STRANDS);
    o2 = strctxsz(0, 8);
    unkc14c += o2;
    s3.unk39c += unkc14c;
    mmio_write(FUC_NEWSCRATCH0, unkc14c);
    strctxsz(0, 3);
    mmwr(0x409418, 1 << myindex, 0, 0);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x400);
}


//e4a unkc10
void unkc10 (cmd,data){
    int o1;
    uint32_t ctx1sz=0;
    int o2;

    mmio_write(FUC_NEWSCRATCH_SET2,0x200);
    // o1 = mmctxsz(3);
    o1 = (mpcount * 0x37 + 0x40)*4;
    o1 = (o1 >> 8) + 1 << 8;
    ctx1sz += o1;
    mmio_write(FUC_NEWSCRATCH4,ctx1sz);
    s3.unk3f8 = mpcount;
    ctx1rsz += o1;
    s3.strands = mmio_read(FUC_STRANDS);
    o2 = strctxsz(0, 3);
    ctx1sz += o2;
    s3.strsz += ctx1sz;

    mmio_write(FUC_NEWSCRATCH0,ctx1sz);
    mmwr(0x409418,1 << myindex, 0,0);
    mmio_write(FUC_NEWSCRATCH_CLEAR2,0x200);

}


// eee unkc11

void unkc11(cmd, data) {
    mmio_write(FUC_NEWSCRATCH_SET2, 0x100);
    mmio_write(0x86c, 0x10);
    mmio_write_i(0x928, 0x3f, 0xc);
    waitstr();
    funka96(0);
    waitstr();
    set93c(0xf, 1);
    waitstr();
    funka40(0);
    waitstr();
    set93c(0xf, 0);
    waitstr();
    set93c(3, 1);
    waitstr();
    mmio_write_i(0x928, 0x3f, 0xd);
    funk47c(8);
    mmio_write(0x86c, 0);
    mmwr(0x409418, 1 << myindex, 0, 0);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x100);
}


// f9b loadctx
int loadctx (uint32_t cmd,uint32_t data,int u1) {
    int i,temp,base;
#if 0
    mmio_write(FUC_NEWSCRATCH_SET2, 0x80000);
    setxferbase(curctxbase);
    if (curs3pres <= 2) {
	funk175(8);
    }
    // fd3
    mmio_write(0x874, 0);
    mmio_write(0x878, 0xb);
    mmio_write(0x8a8, 0x2040000);
    mmio_write(0x86c, 0x10);
    load3(2);
    if (s3.marker != 0xad0becab) {
	// 133d
	error(1);
	mmwr(0x409834, 1 << myindex, 0, 0);
	mmwr(0x409418, 1 << myindex, 0, 0);
	mmio_write(0x424, 0);
	mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80000);
	return 0;
    }
    funk1da(unk18);
    mmio_write_i(0x928, 0x3f, 0xc);
    waitstr();
    mmio_write_i(0x91c, 0x3f, 0);
    mmio_write_i(0x928, 0x3f, 4);
    mmio_write(FUC_MMCTX_LOAD_COUNT, mpcount * 0x37 + 0x40);
    mmio_write(FUC_MMCTX_LOAD_SWBASE, s3.mmbase & 0xffff);
    mmctx(2, 4, 1, mmio_table2, nr_mmio_table2);
    mmctx(2, 3, 4, mmio_table1, nr_mmio_table1);
    waitstr();
    if (curs3pres == 2) {
	setxferbase(curs3base);
	mmio_write_i(0x928, 0x3f, 0xc);
	waitstr();
	set93c(0xf, 0);
	waitstr();
	set93c(8, 1);
	waitstr();
	funka96(0);
	extr(temp,s3.unk3a0, 16,23);
	base = temp + ( s3.unk3a0 & 0xffff);
	for (i = 0; i != s3.unk394; i++) {
	    mmio_write_i(FUC_STRAND_LOAD_SWBASE, i, s3.unk3a4[i].a + base);
	}
	waitstr();
	mmio_write_i(0x91c, 0x3f, 0);
	mmio_write_i(0x928, 0x3f, 4);
	mmio_write(FUC_MMCTX_LOAD_COUNT, 0x38);
	mmio_write(FUC_MMCTX_LOAD_SWBASE, s3.unk3a0 & 0xffff);
	mmctx(2, 5, 2, mmio_table3, nr_mmio_table3);
	waitstr();
	set93c(0xf, 0);
	waitstr();
	set93c(3, 1);
	waitstr();
	funka96(0);
	extr(temp, s3.mmbase, 16, 23);
	base = temp + (s3.mmbase & 0xffff);
	for (i = 0; i != s3.strands; i++) {
	    mmio_write(FUC_STRAND_LOAD_SWBASE, s3.unk314[i].a + base);
	}
	waitstr();
	setxferbase(curctxbase);
    }
    if (curs2pres == 1) {
	setxferbase(curs2base);
	mmio_write(FUC_MMCTX_LOAD_COUNT, mpcount * 4 + 0x3f);
	mmio_write(FUC_MMCTX_LOAD_SWBASE, s3.unk3a4[8].b & 0xffff);
	mmctx(2, 7, 1, mmio_table5, nr_mmio_table5);
	mmctx(2, 6, 3, mmio_table4, nr_mmio_table4);
	mmctx(2, 0xb, 3, mmio_table6, nr_mmio_table6);
	mmctx(2, 0xc, 4, mmio_table7, nr_mmio_table7);
	setxferbase(curctxbase);
    }
    // 12bc
    mmio_write_i(0x928, 0x3f, 0xd);
    waitstr();
    mmwr(0x409418, 1 << myindex, 0, 0);
    mmio_write(0x874, 0);
    mmio_write(0x878, 4);
    mmio_write(0x8a8, 0x7000000);
    mmio_write(0x430, 0);
    mmio_write(0x424, 0);
    mmio_write(FUC_NEWSCRATCH0, 4);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80000);
#endif
    return 0;
}


// 139b savectx

void savectx (uint32_t cmd,uint32_t data,short u1,short u2) {
    int exmmbase,base;	
    int i;
    mmio_write(FUC_NEWSCRATCH_SET2, 0x40000);
    if (u1 == 2) {
	mmio_write(0x874, 0);
	mmio_write(0x878, 9);
	mmio_write(0x8a8, 0x4030000);
    } else {
	// 13db
	mmio_write(0x874, 0);
	mmio_write(0x878, 0xa);
	mmio_write(0x8a8, 0x5020000);
    }
    // 13fc
    funk1da(unk18);
    setxferbase(curctxbase);
    mmio_write_i(0x928, 0x3f, 0xc);
    waitstr();
    mmio_write_i(0x91c, 0x3f, 0);
    mmio_write_i(0x928, 0x3f, 3);
    mmio_write(FUC_MMCTX_SAVE_SWBASE, s3.mmbase & 0xffff);
    mmctx(1, 4, 1, mmio_table2, nr_mmio_table2);
    mmctx(1, 3, 4, mmio_table1, nr_mmio_table1);
    waitstr();
    if (curs3pres == 2) {
	setxferbase(curs3base);
	mmio_write_i(0x928, 0x3f, 0xc);
	waitstr();
	set93c(0xf, 0);
	waitstr();
	set93c(8, 1);
	waitstr();
	funka96(0);
	extr(exmmbase, s3.unk3a0, 16,23);
	base = exmmbase + s3.unk3a0 & 0xffff;
	for (i = 0; i != s3.unk394; i++) {
	    mmio_write_i(FUC_STRAND_SAVE_SWBASE, i, s3.unk3a4[i].a + i * 8 + base);
	}
	waitstr();
	mmio_write_i(0x91c, 0x3f, 0);
	mmio_write_i(0x928, 0x3f, 3);
	mmio_write(FUC_MMCTX_SAVE_SWBASE, s3.unk3a0 & 0xffff);
	mmctx(1, 5, 2, mmio_table3, nr_mmio_table3);//
	waitstr();
	set93c(0xf, 1);
	waitstr();
	set93c(3, 1);
	waitstr();
	funka96(0);
	extr(exmmbase,s3.mmbase,16,23);
	base = exmmbase + s3.mmbase & 0xffff;
	for (i = 0; i != s3.strands; i++) {
	    mmio_write_i(FUC_STRAND_SAVE_SWBASE, i, s3.unk314[i].a + base);
	}
	waitstr();
	setxferbase(curctxbase);
    }
    // 15d8
    if (curs2pres == 1) {
	setxferbase(curs2base);
	mmio_write(FUC_MMCTX_SAVE_SWBASE, s3.unk3a4[8].b & 0xffff);
	mmctx(1, 7, 1, mmio_table5, nr_mmio_table5);
	mmctx(1, 6, 3, mmio_table4, nr_mmio_table4);
	mmctx(1, 0xb, 3, mmio_table6, nr_mmio_table6);
	mmctx(1, 0xc, 4, mmio_table7, nr_mmio_table7);
	setxferbase(curctxbase);
    }
    // 166d
    funk1da(unk18);
    s3.marker = 0xad0becab;
    store3(2);
    waitdone_12();
    mmio_write(0xa10, 1);
    while((mmio_read(0xa10)));
    if (u1 == 0) {
	mmio_write_i(0x928, 0x3f, 0xd);
	waitstr();
	mmwr(0x409418, 1 << myindex, 0, 0);
	mmio_write(0x874, 0);
	mmio_write(0x878, 4);
	mmio_write(0x8a8, 0x2000000);
    } else {
	// 170e
	mmwr(0x409418, 1 << myindex, 0, 0);
    }
    // 1724
    mmio_write(0x430, 0);
    mmio_write(FUC_NEWSCRATCH0, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x40000);
    mmio_write(FUC_NEWSCRATCH5,0x600d);
}


void work(void){

    uint32_t cmd,data;
    mmio_write(FUC_NEWSCRATCH_SET2,0x8);
    if(q2get != q2put){
	//17c7
	cmd = q2[q2get&7].cmd & ~0x7800;
	data = q2[q2get&7].data;
	if(++q2get == 0x10)
	    q2get = 0;
    }else if(qget != qput){
	//1799
	cmd = queue[qget&7].cmd & ~0x7800;
	data = queue[qget&7].data;
	if (++qget == 0x10)
	    qget = 0;
    }else{
	mmio_write(FUC_NEWSCRATCH_CLEAR2,0x8);
    }
    //17f6a
    switch(cmd){

	case 0x21:
	    //
	    unk18 = data;
	    break;
	case 0x10:
	    // 1911
	    unkc10(cmd,data);
	    break;
	case 1:
	    // 18e5
	    savectx(cmd, data, 2, 0);
	    break;
	case 2:
	    //18eb
	    loadctx(cmd, data, 1);
	    break;
	case 8:
	    //18f6
	    savectx(cmd, data,0,0);
	    break;
#if 1
	case 6:
	    //1909
	    unkc6(cmd, data);
	    break;
	case 0x16:
	    //1919
	    unkc16(cmd, data);
	    break;
	case 0x25:
	    //1921
	    unkc25(cmd, data);
	    break;
	case 0x11:
	    // 1929
	    unkc11(cmd, data);
	    break;
	case 0x29:
	    // 1931
	    // //selftest(cmd, data);
	    break;
	case 0x13:
	    // 1939
	    unkc13(cmd, data); // inline
	    break;
	case 0x17:
	    // 1944
	    unkc17(cmd, data); // inline
	    break;
	case 0x26:
	    // 194b
	    unkc26(cmd, data); // inline
	    break;
	case 0x19:
	    // 1952
	    unkc19(cmd, data); // inline
	    break;
	case 0x28:
	    // 1959
	    unkc28(cmd, data); // inline
	    break;
	case 0x14:
	    // 1960
	    unkc14(cmd, data);
	    break;
	case 0x18:
	    // 1967
	    unkc18(cmd, data);
	    break;
	case 0x27:
	    // 196e
	    unkc27(cmd, data);
	    break;
	case 0x12:
	    // 1975
	    done = 1;
	    break;
	case 0:
	case 0x15:
	case 0x10000:
	case 0x10001:
	    break;
#endif
	default:
	    // 1982
	    err2(0x11);
	    break; 
    }
    // 1989
    mmio_write(FUC_NEWSCRATCH_CLEAR2,8);
    return;
}




/*
 * entry point into the microcontroller code.
 */
int main(void)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 1);
    done = 0;
    mmio_write(0x048, 2);
    //ihandler = &&ih;
    mmio_write(0x01c, 0);
    mmio_write(0x010, 0x5004);
    mmio_write(0x00c, 4);
    mmio_write_i(0x404, 6, 0x202d);
    mmio_write_i(0x404, 5, 0x202b);
    mmio_write_i(0x404, 4, 0x2002);
    set_flags(ie0);
    set_flags($p2);
    myindex = mmio_read(FUC_MYINDEX) & 0x1f;
    mpcount = mmio_read(FUC_UNITS) & 0x1f;

    mmio_write(FUC_MMCTX_SAVE_SWBASE, 0);
    mmio_write(FUC_MMCTX_LOAD_SWBASE, 0);
    mmio_write(FUC_NEWSCRATCH0, 0x87654321);
    while (1) {
	// 1a56
	sleep($p2);
	work();
	if (done)
	    break;
	set_flags($p2);
	if (qput != qget)
	    clr_flags($p2);
    }

    // 1a87
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 1);
    return 0;
}
