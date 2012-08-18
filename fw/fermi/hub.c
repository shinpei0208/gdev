#include "asminsn.h"
#include "mmio.h"
#include "regs_hub.h"
#include "types.h"

// 0..18 stamp
int stamp[] = {
    0x2072614d,
    0x32203120,
    0x00303130,
    0x323a3031,
    0x32333a39,
    0x00000000
};

// 18
int unk18 = 0x186a0;
// 1c
int testres = 0x1000;
// 20
int unk20[11] = {
    1, 1, 2, 2, 1, 1, 7, 1, 1, 1, 2
};

//int temp[4]={0x123,0x234,0x345,0x456};

// 4c..18c mmio tables
int mmio_table1[] = {
    0x0417e91c, // #0
    0x04400204, // #1
    0x28404004, // #2
    0x00404044, // #3
    0x34404094, // #4
    0x184040d0, // #5
    0x004040f8, // #6
    0x08404130, // #7
    0x08404150, // #8
    0x04404164, // #9
    0x08404174, // #10
    0x1c404200, // #11
    0x34404404, // #12
    0x0c404460, // #13
    0x00404480, // #14
    0x00404498, // #15
    0x0c404604, // #16
    0x7c404618, // #17
    0x50404698, // #18
    0x044046f0, // #19
    0x54404700, // #20
    0x00405800, // #21
    0x08405830, // #22
    0x00405854, // #23
    0x0c405870, // #24
    0x04405a00, // #25
    0x00405a18, // #26
    0x00406020, // #27
    0x0c406028, // #28
    0x044064a8, // #29
    0x044064b4, // #30
    0x00407804, // #31
    0x1440780c, // #32
    0x004078bc, // #33
    0x18408000, // #34
    0x00408064, // #35
    0x08408800, // #36
    0x0c408900, // #37
    0x00408980  // #38
};
int nr_mmio_table1 = 0x27; // 39

int mmio_table2[] = {
    0x0040415c, // #0
    0x00407808, // #1
    0x1c408914, // #2
    0x04408988, // #3
    0x7c1b0040, // #4
    0x401b00c0, // #5
    0x7c1b0240, // #6
    0x401b02c0, // #7
    0x7c1b0440, // #8
    0x401b04c0, // #9
    0x7c1b0640, // #10
    0x401b06c0, // #11
    0x7c1b0840, // #12
    0x401b08c0, // #13
    0x7c1b0a40, // #14
    0x401b0ac0, // #15
    0x7c1b0c40, // #16
    0x401b0cc0, // #17
    0x7c1b0e40, // #18
    0x401b0ec0, // #19
    0x081b4000, // #20
    0x081b4010, // #21
    0x081b4020, // #22
    0x081b4030, // #23
    0x081b4040, // #24
    0x081b4050, // #25
    0x0c1b4060, // #26
    0x081b4074, // #27
    0x0c1b4090, // #28
    0x001b40a4, // #29
    0x141b4100, // #30
    0x141be000  // #31
};
int nr_mmio_table2 = 0x20;

int mmio_table3[] = {
    0x7c000040, // #0
    0x400000c0, // #1
    0x7c000240, // #2
    0x400002c0  // #3
};
int nr_mmio_table3 = 0x4;

int mmio_table4[] = {
    0x14000000 // #0
};
int nr_mmio_table4 = 0x1;

struct pair 
{
    int a, b;
};

struct qe {
    uint32_t cmd;
    uint32_t data;
};

// 18c
int done;
// 190
int unk190;
// 194
int unk194;
// 198
int mmsz;
// 19c
int unk19c; /* TP grctx size? */
// 1a0... never seen used
int unk1a0;
// 1a4
int unk1a4;
// 1a8
int unk1a8;
// 1ac
int fakeautosw;
// 1b0
int curctxbase;
// 1b4
int ach_ctxbase;
// 1b8
int ach_chan;
// 1bc
int ach;
// 1c0
int ach_s5;
// 1c4
int curs3base;
// 1c8
int curs2base;
// 1cc
int curs3pres;
// 1d0
int curs2pres;
// 1d4
int tpcount;
// 1d8
int ropcount;
// 1dc
struct qe queue[8];
// 21c
int qget;
// 220
int qput;
// 224
struct qe q2[8];
// 264
int q2get=0;
// 268
int q2put=0;
// 26c..300 unk
int unk26c;
int unk270;

struct s_temp{
    int a;
} s_temp __attribute__ ((aligned (0x100))) = {0};


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
    // 314..358 unk
    struct pair unk314[9];
    // 35c..3a4 unk
    struct pair unk35c_3a4[9];//add
    // 3a4..3e8 unk
    struct pair unk3a4[9];
    // 3ec
    int unk3ec;
    // 3f0
    int unk3f0;
    // 3f4
    int unk3f4;
    //3f8
    int unk3f8;
    // 3fc
    int marker
} s3 __attribute__ ((aligned (0x100))) = {0};

// 400
struct s4 
{
    struct pair unk400;
    struct pair unk408;    
    struct pair unk410;
    struct pair unk418[5];
} s4 __attribute__ ((aligned (0x100))) = {0};

// 500
volatile struct s5 {
    int unk500;
    int unk504;
    int tpcount; // 508
    int unk50c;
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
    // 540...5f0 unk
    int unk540[45];
    int unk5f4;
    int unk5f8;
    int marker // 5fc
} s5 __attribute__ ((aligned (0x100))) = {0};

volatile void test_write(int value){
#if 1
    mmio_write(FUC_NEWSCRATCH4,value);
#endif
}



int mmctxsz(int x)
{
    int res;

    switch (x) {
	case 1:
	    res = 0x32c;
	    break;
	case 2:
	    res = 0x6f8;
	    break;
	case 9:
	    // 36
	    res = 0x18 * ropcount + (ropcount * 0x62 << 2);
	    break;
	default:
	    res = 0;
	    break;
    }

    return ((res >> 8) + 1) << 8;
}

#if 0
void funk64(int x) // never used
{
    mmio_write(0x614, 0x20);
    unk270 = x;
    while (--x);
    unk270 = x;
    mmio_write(0x614, 0xa20);
}
#endif

void funk9e (int x)
{
    // always inlined
    mmio_write_i(FUC_STRAND_FIRST_GENE, 0x3f, x);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 3);
}

void xfer(int base, int offset, int sizel2, int local_addr,
	int subtrg, int trg, int dir, int wait)
{
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

void store5(int wait)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 8);
    xfer(0, 0, 6, 0x500, 0, 2, 2, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 8);

}

void store3(int wait)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 2);
    xfer(0, 0x100, 6, 0x300, 0, 2, 2, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 2);
}

void load3 (int wait)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 1);
    xfer(0, 0x100, 6, 0x300, 0, 2, 1, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 1);
}

void load5 (int wait)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 4);
    xfer(0, 0, 6, 0x500, 0, 2, 1, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 4);
}

void setfakeautosw(uint32_t cmd, uint32_t data) // always inlined
{
    fakeautosw = 1;
    mmio_write(FUC_NEWSCRATCH0, 1);
}

#if 0
void funk28e(int x) // never used
{
    if (x)
	mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xc);
    else
	mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xd);
}
#endif

void set93c(int x, int y)
{
    mmio_write_i(0x93c, 0x3f, x);
    if (y) 
	mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xa);
    else
	mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xb);
}

#if 0
void mmwrq(uint32_t addr, int val, int z) // never used
{
    mmio_write(FUC_MMIO_WRVAL, val);
    mmio_write(FUC_MMIO_CTRL, (addr & 0x3fffffc) | (z & 1) | 0xc0000000);
}

void funk304 (void) // never used
{
    mmio_write(0x430, 0);
}
#endif

void funk312(int x)//watchdoc timer set
{
    if (x){
	mmio_write(0x430, (0x7fffffff & 0x3fffffff) | 0x80000000);
    }
}

#if 0
void funk338 (int x) // never used
{
    mmio_write_i(0x91c, 0x3f, x);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 4);
}

void funk351 (int x, int y, int z) // never used
{
    mmio_write(0x874, y);
    mmio_write(0x878, x & 0xf);
    mmio_write(0x8a8, z);
}

int funk387 (int x) // never used
{
    int i;

    for (i = 0; i != 9; i++) {
	int val = mmio_read_i(FUC_STRAND_SIZE, i);
	if (val > 0) {
	    int xval;
	    mmio_write_i(FUC_STRAND_SAVE_SWBASE, i, x);
	    mmio_write_i(FUC_STRAND_LOAD_SWBASE, i, x);
	    extrs(xval, val, 6, 29);
	    x += xval + 1;	    
	}
    }
    return x;
}
#endif

#if 0 // hmm, let's hope the test always succeeds...
int scratchtest(int reg, int clear, int set)
{
    // args are really shifted by 6 as I[] addresses, but meh
    int saved = mmio_read(reg);

    mmio_write(clear, 0xffffffff);
    if (mmio_read(reg))
	goto out;
    mmio_write(set, 0x55555555);
    if (mmio_read(reg) != 0x55555555)
	goto out;
    mmio_write(clear, 0);
    if (mmio_read(reg) != 0x55555555)
	goto out;
    mmio_write(reg, 0xaaaaaaaa);
    if (mmio_read(reg) != 0xaaaaaaaa)
	goto out;
    mmio_write(clear, 0xffffffff);
    if (mmio_read(reg))
	goto out;
    mmio_write(reg, saved);
    return 0x1000;

out:
    mmio_write(reg, saved);
    return 0x2000;
}
#endif

void waitstr(void)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x4000);
    while ((mmio_read(FUC_DONE) & 4));
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x4000);
}

void funk486(int x)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x8000);
    mmio_write_i(0x91c, 0x3f, x);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 1);
    waitstr();
    mmio_write_i(0x918, 0x3f, 0);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 5);
    waitstr();
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x8000);
}

void funk4dc(int x)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x2000);
    mmio_write_i(0x91c, 0x3f, x);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 1);
    waitstr();
    mmio_write_i(0x918, 0x3f, -0x1);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 2);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x2000);
}

void mmwr(uint32_t addr, int val, uint32_t u1, uint32_t u2)
{
    extr(addr, addr, 2, 25);
    mmio_write(FUC_MMIO_WRVAL, val);
    mmio_write(FUC_MMIO_CTRL, (addr << 2) |
	    (u1 & 1) | 0xc0000000 | (!!u2) << 29);
    while (mmio_read(FUC_MMIO_CTRL) & 0x80000000);
    if (u2)
	while (!(mmio_read(FUC_DONE)&0x80));
}

void funk598(int x)
{
    mmio_write(0x614, 0x270);
    mmwr(0x41a614, 0x820, 0, 1); // XXX mmwr with 1
    unk26c = x;
    while(--x);
    unk26c = x;
    mmio_write(0x614, 0x770);
    mmwr(0x41a614, 0xa20, 0, 1); // XXX mmwr with 1
}

void unkc6(uint32_t cmd, uint32_t data)
{
    funk598(8);
    mmio_write(FUC_NEWSCRATCH0, 0x100);
}

int mmrd (uint32_t addr, uint32_t u1)
{
    extr(addr, addr, 2, 25);
    addr = addr << 2;
    mmio_write(FUC_MMIO_CTRL, addr  | (u1 & 1) | 0x80000000);
    while (mmio_read(FUC_MMIO_CTRL) & 0x80000000);
    while (!(mmio_read(FUC_DONE) & 0x40));
    return mmio_read(FUC_MMIO_RDVAL);
}

void wait4170 (void)
{
    while (mmrd(0x404170, 0) & 0x10);
}

void set4170 (int x, int wait)
{
    mmwr(0x404170, (x & 3) | 0x10, 0, 1); // XXX mmwr with 1
    if (wait == 1)
	wait4170(); // inlined
}

void set4170_0_w(void)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x40000);
    set4170(0, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x40000);
}

void unkc5 (uint32_t cmd, uint32_t data)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 0x8000);
    set4170_0_w();
    mmwr(0x409614, 0x770, 0, 1); // XXX mmwr with 1
    mmwr(0x41a614, 0xa20, 0, 1); // XXX mmwr with 1
    mmwr(0x408a10, 0x440, 0, 1); // XXX mmwr with 1
    mmio_write(FUC_NEWSCRATCH0, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x8000);
}

void set4170_0(int wait)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x80);
    set4170(0, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x80);
}

void set4170_1_w(void)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x20000);
    set4170(1, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x20000);
}

void set4170_2(int wait)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x10000);
    set4170(2, wait);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x10000);
}

#if 0
void set4170_3_w(void) // never used
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x40);
    set4170(3, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x40);
}
#endif

void set4160(void)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x400);
    mmwr(0x404160, 1, 0, 0);
    while (!(mmrd(0x404160, 0) & 0x10));
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x400);
}

int m34rd (int x, uint32_t class)
{
    uint32_t req, xval;

    switch (class & 0xf0ff) {
	case 0x9039:
	case 0x902d:
	case 0x9097:
	case 0x90c0:
	    extr(xval, 0x3400 + x * 4, 2, 13);
	    req = xval << 0x10 | (class & 0xffff) | 0x40000000;
	default:
	    req = 0;
    }
    mmwr(0x404488, req, 0, 0);
    while (mmrd(0x404488, 0) & 0x40000000);

    return mmrd(0x40448c, 0);
}

void fwm4(uint32_t data, uint32_t class)
{
    uint32_t val, mask, old;

    val = m34rd(1, class);
    mask = m34rd(2, class);
    old = mmrd(data, 0);
    mmwr(data, (old & ~mask) | (val & mask), 0, 0);
}

void fwm1(uint32_t data, uint32_t class)
{
    mmwr(data, m34rd(1, class), 0, 0);
}

void m34wr(int x, uint32_t data, uint32_t class)
{
    uint32_t req, xval;

    switch (class & 0xf0ff) {
	case 0x9039:
	case 0x902d:
	case 0x9097:
	case 0x90c0:
	    extr(xval, 0x3400 + x * 4, 2, 13);
	    req = xval << 0x10 | (class & 0xffff) | 0x80000000;
	    break;
	default:
	    req = 0;
	    break;
    }
    mmwr(0x40448c, /*y*/data, 0, 0); // what is y? guess y = data?
    mmwr(0x404488, req, 0, 0);
    while (mmrd(0x404488, 0) & 0x80000000);
}

void fwm0(uint32_t data, uint32_t class)
{
    m34wr(1, mmrd(data, 0), class);
}

void unkc16(uint32_t cmd, uint32_t data)
{
    int i;
    int loc[8];
    int total;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x400);
    mmwr(0x41a800, 0, 0, 0);
    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a500, 0, 0, 0);
    mmwr(0x41a504, 0x16, 0, 0);

    while (!(mmio_read(FUC_DONE)&0x100));

    for (i = 0; i < tpcount; i++) {
	loc[i] = mmrd(0x502800 + i * 0x8000, 0);
	unk19c += loc[i];
    }

    mmwr(0x41a800, 0, 0, 0);
    mmio_write(FUC_BAR_STATUS, 0); // what is BAR_STATUS?

    total = 0;
    for (i = 0; i < tpcount; i++) {
	mmwr(0x502500 + i * 0x8000, total, 0, 0);
	mmwr(0x502504 + i * 0x8000, 0x18, 0, 0);
	total += loc[i] >> 8;
    }

    if (data >= unk19c)
	unk19c = data;
    s5.unk524 = unk19c;
    mmio_write(FUC_NEWSCRATCH0, s5.unk524);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x400);
}

void unkc25(uint32_t cmd, uint32_t data)
{
    int i;
    int o2, o9, ctr, st;
    int xo2, xo9;
    int loc[8];

    mmio_write(FUC_NEWSCRATCH_SET2, 0x800);
    o9 = mmctxsz(9);
    o2 = mmctxsz(2);
    unk1a8 += o2 + o9;
    unk1a4 += o2 + o9;
    extr(xo2, o2, 8, 23); // not sure
    extr(xo9, o9, 8, 15); // not sure
    s3.unk3ec = xo2 | (xo9 << 0x10);
    s3.unk3a4[8].a += unk1a4;
    extr(xo2, o2, 8, 15); // not sure
    s3.unk3a4[8].b = xo2 << 0x10;
    st = unk1a4;
    mmwr(0x41a800, 0, 0, 0);

    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a500, 0, 0, 0);
    mmwr(0x41a504, 0x25, 0, 0);

    while (!(mmio_read(FUC_DONE)&0x100));

    for (i = 0; i < tpcount; i++) {
	loc[i] = mmrd(0x502800 + i * 0x8000, 0);
	unk1a4 += loc[i];
    }

    mmwr(0x41a800, 0, 0, 0);
    ctr = st >> 8;
    mmio_write(FUC_BAR_STATUS, 0);

    for (i = 0; i < tpcount; i++) {
	mmwr(0x502500 + i * 0x8000, ctr, 0, 0);
	mmwr(0x502504 + i * 0x8000, 0x27, 0, 0);
	ctr += loc[i] >> 8;
    }

    s5.unk530 = unk1a4;
    mmio_write(FUC_NEWSCRATCH0, unk1a4);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x800);
}

void waitdone_12(void)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x100);
    while (mmio_read(FUC_DONE) & 0x1000);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x100);
}

void clear4160(void)
{
    mmio_write(FUC_NEWSCRATCH_SET3, 0x800);
    waitdone_12();
    mmwr(0x404160, 0, 0, 0);
    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x800);
}

void setxferbase(int x)
{
    waitdone_12();
    mmio_write(FUC_MEM_BASE, x);
}

void d76 (uint32_t cmd, uint32_t data, int x)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 0x20);
    if ((data	& 0x80000000)) {
	if (x == 1)
	    set4160();

	waitdone_12();
	mmio_write(0xa24, 0);
	mmio_write(0xb04, data);
	mmio_write(FUC_MEM_CHAN, data); /* type=nv50_channnel????? */
	mmio_write(FUC_MEM_CMD, 7);
	while (mmio_read(FUC_MEM_CMD)&0x1f);
	mmio_write(0xb00, data);
	if (x == 1) {
	    uint32_t xdata;
	    setxferbase(0);
	    extr(xdata, data, 28, 29);
	    xfer(data << 4, 0x200, 3, 0x400, xdata, 1, 1, 2);
	    curctxbase = ((s4.unk410.a >> 8) & 0x00ffffff ) | (s4.unk410.b << 0x18);

	    setxferbase(curctxbase);
	    s5.marker = 0x600dc0de;
	    store5(2);
	    waitdone_12();

	    mmio_write(FUC_MEM_CMD, 5);
	    mmwr(BC_BASE + FUC_WRDATA, curctxbase, 0, 0);
	    mmwr(BC_BASE + FUC_WRCMD, 0x13, 0, 0);
	    while(mmio_read(FUC_MEM_CMD));

	    clear4160();
	}
    }

    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x20);
}

#if 0
int xdtest(void)
{
    int i;
    int *ptr400 = (int*)0x400; // s4

    set4160();
    for (i = 0; i < 0x40; i++) {
	ptr400[i] = 0x80000000 | i * 4;
    }

    setxferbase(curctxbase);
    xfer(0, 0x100, 2, 0x400, 0, 2, 2, 1);
    xfer(0, 0x110, 2, 0x410, 0, 2, 2, 1);
    xfer(0, 0x120, 3, 0x420, 0, 2, 2, 1);
    xfer(0, 0x140, 4, 0x440, 0, 2, 2, 1);
    xfer(0, 0x180, 5, 0x480, 0, 2, 2, 2);
    for (i = 0; i < 0x40; i++) {
	if (ptr400[i] != (0x80000000 | i * 4)) {
	    clear4160();
	    return 0x2000;
	}
    }

    for (i = 0; i < 0x40; i++)
	ptr400[i] = 0;

    xfer(0, 0x100, 2, 0x400, 0, 2, 1, 1);
    xfer(0, 0x110, 2, 0x410, 0, 2, 1, 1);
    xfer(0, 0x120, 3, 0x420, 0, 2, 1, 1);
    xfer(0, 0x140, 4, 0x440, 0, 2, 1, 1);
    xfer(0, 0x180, 5, 0x480, 0, 2, 1, 2);
    for (i = 0; i < 0x40; i++) {
	if (ptr400[i] != (0x80000000 | i * 4)) {
	    clear4160();
	    return 0x2000;
	}
    }
    clear4160();
    return 0x1000;
}
#endif
#if 0
void selftest (uint32_t cmd, uint32_t data)
{
    int i, x;

    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a500, data, 0, 0);
    mmwr(0x41a504, 0x29, 0, 0);

    while (!(mmio_read(FUC_DONE)&0x100));

    for (i = 0; i < tpcount; i++) {
	x = mmrd(0x502800 + 0x8000 * i, 0);
	if (x != 0x1000) {
	    testres = 0x2000;
	    return;
	}
    }

    for (i = 0; i < 8; i++) {
	testres = scratchtest(FUC_NEWSCRATCH_BASE + 4 * i,
		FUC_NEWSCRATCH_SET_BASE + 4 * i,
		FUC_NEWSCRATCH_CLEAR_BASE + 4 * i);
	if (testres == 0x2000)
	    return;
    }

    testres = xdtest();
}
#endif

void fwm2(uint32_t data, uint32_t class)
{
    int m1, m2;

    m1 = m34rd(1, class);
    m2 = m34rd(2, class);

    set4160();
    waitdone_12();
    mmio_write(0xa24, m1);
    mmio_write(FUC_MEM_CHAN, m2);
    mmio_write(0xa10, data);

    while (mmio_read(0xa10));

    clear4160();
}

void unkc4(uint32_t cmd, uint32_t data)
{
    int i;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x4000);
    set4170_2(1);
    mmwr(HUB_BASE + FUC_RED_SWITCH, 0x771, 0, 1); // XXX mmwr with 1
    mmwr(BC_BASE + FUC_RED_SWITCH, 0xa22, 0, 1); // XXX mmwr with 1
    mmwr(0x408a10, 0x444, 0, 1); // XXX mmwr with 1

    mmio_write(0xc08, 8);

    for (i = 0; i != data; i++) {
	if ((mmio_read(FUC_TPCONF) & 0x8000)) {//chang 0xc00TPCONF
	    mmio_write(FUC_NEWSCRATCH0, 1);
	    goto done;
	}
    }
    mmio_write(FUC_NEWSCRATCH0, 2);

done:
    set4170_1_w();
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x4000);
}

void unkcb(uint32_t cmd, uint32_t data)
{
    int i;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x80);
    mmio_write(DEPTH_RANGE_NEAR, 4);
    for (i = 0; i != data; i++) {
	if ((mmio_read(FUC_TPCONF) & 0x2000)) {
	    mmio_write(FUC_NEWSCRATCH0, 0x5);
	    goto done;
	}
    }
    mmio_write(FUC_NEWSCRATCH0, 0xa);

done:
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80);
}

void mmctx(int dir, int y, int z, int *ptraddr, int len)
{
    int i;
    int mode, free;
    int *ptr;
    ptr = ptraddr;
    mmio_write(FUC_NEWSCRATCH_SET3, 0x10);
    switch (y) {
	case 9:
	    mmio_write(FUC_MMCTX_BASE, 0x1a0000);
	    mmio_write(FUC_MMCTX_MULTI_STRIDE, 0x1000);
	    mmio_write(FUC_MMCTX_MULTI_MASK, (1 << ropcount) - 1);
	    mode = 3;
	    break;
	case 0xa:
	    mmio_write(FUC_MMCTX_BASE, 0x1bc000);
	    mmio_write(FUC_MMCTX_MULTI_STRIDE, 0x200);
	    mmio_write(FUC_MMCTX_MULTI_MASK, (1 << ropcount) - 1);
	    mode = 3;
	    break;
	default:
	    mode = 0;
    }

    if (z <= 2)
	mmio_write(FUC_MMCTX_CTRL, ((dir == 2) << 0x10) | 0x21000);

    free = 0;
    for (i = 0; i < len; i++) {
	while (!free)
	    free = mmio_read(FUC_MMCTX_CTRL) & 0x1f;
	mmio_write(FUC_MMCTX_QUEUE, ptr[i] | mode);
	free--;
    }

    if (z == 1 || z == 3) {
	while (free != 0x10)
	    free = mmio_read(FUC_MMCTX_CTRL) & 0x1f;

	while (!(mmio_read(FUC_DONE)&0x20));
    } else {
	mmio_write(FUC_MMCTX_CTRL, ((dir == 2) << 0x10) | 0x41000);
	while (mmio_read(FUC_MMCTX_CTRL) & 0x40000);
    }

    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x10);
}

void unkc15 (uint32_t cmd, uint32_t data)
{
    mmio_write(0x874, data & 0x3fffffff);
    mmio_write(0x878, 0xc);
    mmio_write(0x8a8, 0x3040000);
    mmio_write(0x874, 0);
    mmio_write(0x878, 4);
    mmio_write(0x8a8, 0x7000000);
    mmio_write(FUC_NEWSCRATCH0, 1);
}

void unkc11(uint32_t cmd, uint32_t data)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 0x100);
    mmwr(0x41a800, 0, 0, 0);
    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a500, 0, 0, 0);
    mmwr(0x41a504, 0x11, 0, 0);
    mmwr(0x408a14, 0x10, 0, 0);
    mmio_write(0x86c, 0x10);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xc);
    waitstr();
    funk4dc(0);
    waitstr();
    set93c(0xf, 1);
    waitstr();
    funk486(0);
    waitstr();
    set93c(0xf, 0);
    waitstr();
    set93c(3, 1);
    waitstr();
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xd);
    funk598(8);
    mmio_write(0x86c, 0);
    mmwr(0x408a14, 0, 0, 0);

    while (!(mmio_read(FUC_DONE)&0x100));

    mmio_write(FUC_NEWSCRATCH0, 1);
    mmio_write(0x424, 0); //exit code???
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x100);
}

int strctxsz(int x, int y)
{
    int i, res;
    int rr, ss;

    mmio_write(FUC_NEWSCRATCH_SET3, 0x1000);

    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xc); 
    waitstr();
    set93c(0xf, 0);
    waitstr();
    set93c(y, 1);
    waitstr();
    funk4dc(0);
    waitstr();
    set93c(0xf, 0);
    waitstr();
    set93c(3, 1);
    waitstr();
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xd);
    waitstr();

    res = 0;
    for (i = 0; i != 9; i++) {
	rr = mmio_read_i(FUC_STRAND_SIZE, i);
	if (rr > 0) {
	    rr <<= 2;
	    if (y == 8) {
		s3.unk3a4[i].a = x;
		s3.unk3a4[i].b = rr;
	    } else {
		s3.unk314[i].a = x;
		s3.unk314[i].b = rr;
	    }
	    ss = (rr >> 8) + 1;
	    x += ss;
	    res += ss << 8;
	}
    }

    mmio_write(FUC_NEWSCRATCH_CLEAR3, 0x1000);
    return res;
}

void unkc10(uint32_t cmd, uint32_t data)
{
    int i;
    int loc[8];
    int o1, r0, r10, st, a, x;
    int xmmsz, xmmbase;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x200);
    o1 = mmctxsz(1);
    unk194 += o1;
    mmsz += o1;
    s3.strands = mmio_read(FUC_STRANDS);
    r10 = strctxsz(0, 3) + unk194;
    s3.strsz += r10;

    mmio_write_i(FUC_UNKA00, 0,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 1,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 2,((2 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 3,((2 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 4,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 5,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 6,((7 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 7,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 8,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 9,((1 & 0x7) << 4) | 8);
    mmio_write_i(FUC_UNKA00, 0xa,((2 & 0x7) << 4) | 8);

    st = unk194 = (tpcount << 8) + r10 + 0x200;
    extr(xmmsz, mmsz, 8, 15);
    s3.mmbase = ((tpcount + 2) & 0xffff) | (xmmsz << 0x10);
    mmwr(0x41a800, 0, 0, 0);

    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a500, 0, 0, 0);
    mmwr(0x41a504, 0x10, 0, 0);

    while (!(mmio_read(FUC_DONE)&0x100));
    for (i = 0; i < tpcount; i++) {
	loc[i] = mmrd(0x502800 + i * 0x8000, 0);
	unk194 += loc[i];
    }

    mmwr(0x41a800, 0, 0, 0);
    r0 = sar(st,0x8);
    mmio_write(FUC_BAR_STATUS, 0);
    for (i = 0; i < tpcount; i++) {
	mmwr(0x502500 + i * 0x8000, r0, 0, 0);
	mmwr(0x502504 + i * 0x8000, 0x14, 0, 0);
	r0 += (loc[i] >> 8);
    }

    if (data > unk194)
	unk194 = data;
    
    s5.tpcount = tpcount;
    st = s5.unk504 = unk194;
    extr(xmmbase, s3.mmbase, 16, 23);
 
    a = xmmbase + s3.mmbase & 0xffff;
    for (i = 0; i != s3.strands; i++) {
	x = s3.unk314[i].a + a;
	mmio_write_i(FUC_STRAND_SAVE_SWBASE, i, x);
	mmio_write_i(FUC_STRAND_LOAD_SWBASE, i, x);
    }

    mmio_write(FUC_NEWSCRATCH0, st);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x200);

}

void error(int ecode)
{
    // always inlined... and duplicated with 36e!
    mmio_write(FUC_NEWSCRATCH6, ecode);//add
    mmio_write(FUC_INTR_UP_SET, 1);
}

void enqueue(uint32_t cmd, uint32_t data)
{
    struct qe *ptr;

    mmio_write(FUC_NEWSCRATCH_SET2, 2);
    ptr = &queue[qput & 7];
    if ((qget & 7) == (qput & 7) && (qget & 8) != (qput & 8)) {
	error(0x12); // inline
    } else {
	ptr->data = data;
	ptr->cmd = cmd;
	if (++qput == 0x10)
	    qput = 0;
    }

    mmio_write(FUC_NEWSCRATCH_CLEAR2, 2);
}

void ihbody(void)
{

    int intr = mmio_read(FUC_INTR);
    if (intr & 4) { //fix this
	enqueue(mmio_read(0x068), mmio_read(0x064));
	mmio_write(0x074, 1);
    }
    if (intr & 0xff00) { //fix this
	if (intr & 0x100) { // auto ctx switch
	    enqueue(0x10000, 0);
	}
	if (intr & 0x200) {//fix this
	    enqueue(4, 0);
	}
	if (intr & 0x400) { // fwmthd fix this
	    enqueue(0x10002, 0);
	}
	if (intr & 0x8000) { // nop?!? fix this
	    enqueue(0x10003, 0);
	}
    }
    mmio_write(0x004, intr);
}

void fwm5(uint32_t data, uint32_t class)
{
    int r1;
    int val, mask;
    int *ptr;
    uint32_t xdata;

    val = m34rd(1, class);
    mask = m34rd(2, class);

    extr(xdata, data, 16, 19);
    switch (xdata) {
	case 0:
	    ach_s5 = 0;
	    r1 = 0;
	    break;
	case 1:
	    r1 = ((data >> 0x14) << 8) + 0x100;
	    break;
	default:
	    error(0x13); // inline
	    break;
    }

    set4160();
    xfer(0, r1, 6, 0x400, 0, 2, 1, 2);
    ptr = (int*)(0x400 + (data & 0xfffc));
    *ptr = (mask & val) | (~mask & *ptr);
    xfer(0, r1, 6, 0x400, 0, 2, 2, 2);
    clear4160();
}

void fwmthd (uint32_t cmd, uint32_t data)
{
    uint32_t addr, mthd, subc, ctx, class;
    uint32_t xaddr;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x2000);
    addr = mmrd(0x400704, 0);
    mthd = addr & 0x3ffc;
    extr(xaddr, addr, 16, 18);
    subc = xaddr;
    ctx = mmrd(0x404200 + subc * 4, 0);
    class = ctx & 0xffff;
    data = mmrd(0x400708, 0); // overwritten???

    while (mmrd(0x400700, 0) & 0x0303fb7c);

    switch (class & 0xf0ff) {
	case 0x9097:
	    switch (mthd) {
		case 0x2300:
		    goto m0;
		case 0x2304:
		    goto m1;
		case 0x2308:
		    goto m2;
		case 0x230c:
		    goto m3;
		case 0x2310:
		    goto m4;
		case 0x2314:
		    goto m5;
		default:
		    goto bad;
	    }
	case 0x90c0:
	    switch (mthd) {
		case 0x500:
		    goto m0;
		case 0x504:
		    goto m1;
		case 0x508:
		    goto m2;
		case 0x50c:
		    goto m3;
		case 0x510:
		    goto m4;
		case 0x514:
		    goto m5;
		default:
		    goto bad;
	    }
	case 0x9039:
	    switch (mthd) {
		case 0x27c:
		    goto m0;
		case 0x280:
		    goto m1;
		case 0x284:
		    goto m2;
		case 0x288:
		    goto m3;
		case 0x28c:
		    goto m4;
		case 0x290:
		    goto m5;
		default:
		    goto bad;
	    }
	case 0x902d:
	    switch (mthd) {
		case 0x8e0:
		    goto m0;
		case 0x8e4:
		    goto m1;
		case 0x8e8:
		    goto m2;
		case 0x8ec:
		    goto m3;
		case 0x8f0:
		    goto m4;
		case 0x8f4:
		    goto m5;
		default:
		    goto bad;
	    }
	default:
	    goto bad;
    }
m0:
    fwm0(data, class);
    goto ok;
m1:
    fwm1(data, class);
    goto ok;
m2:
    fwm2(data, class);
    goto ok;
m3:
    //selftest(0x29, data);
    m34wr(1, testres, class);
    goto ok;
m4:
    fwm4(data, class);
    goto ok;
m5:
    fwm5(data, class);
    goto ok;
bad:
    m34wr(0, 1, class);
    mmwr(0x400144, 0x100, 0, 0);
    mmio_write(0xc1c, 0x20000);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x2000);
    return;
ok:
    m34wr(0, 1, class);
    mmwr(0x400144, 0x100, 0, 0);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x2000);
}

void enq2(uint32_t cmd, uint32_t data)
{
    uint32_t xq2get, xq2put;

    mmio_write(FUC_NEWSCRATCH_SET2, 4);
    extr(xq2get, q2get, 3, 3);
    extr(xq2put, q2put, 3, 3);    
    if ((q2put & 7) == (q2get & 7) && xq2get != xq2put) {
	error(0x12); // inline
    } else {
	q2[q2put&7].cmd = cmd;
	q2[q2put&7].data = data;
	if (++q2put == 0x10)
	    q2put = 0;
    }
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 4);
}

void qwork(void)
{
    uint32_t cmd, data;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x10);
    if (qget != qput) {
	cmd = queue[qget&7].cmd & ~0x7800;
	data = queue[qget&7].data;
	if (++qget == 0x10)
	    qget = 0;
	switch (cmd) {
	    case 4:
		unkc4(cmd, data);
		break;
	    case 5:
		unkc5(cmd, data);
		mmio_write(0xc08, 4);
		break;
	    case 0x10002:
		fwmthd(cmd, data);
		break;
	    case 0x10003:
		break;
	    default:
		enq2(cmd, data);
		break;
	}
    }
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x10);
}

void funk1f47(int dowork)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 0x20);
    mmio_write(0xc08, 4);
    while ((mmio_read(0xc00)&0x2000)) {
	if (dowork)
	    qwork();
    }
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x20);
}

int loadctx (uint32_t cmd, uint32_t data, uint32_t u1)
{
    int i, j;
    int mmcnt, ctxbase, tm;
    uint32_t xdata;
    mmio_write(FUC_NEWSCRATCH_SET2, 0x80000);
#if 1
    set4170_2(0);
    if (u1 == 1) {
	funk1f47(u1);
	mmio_write(0x86c, 0x10);
	mmwr(0x408a14, 0x10, 0, 0);
	mmwr(0x41a86c, 0x10, 0, 0);
    }

    funk312(unk18);
    mmio_write(0x874, data & 0x3fffffff);
    mmio_write(0x878, 0xb);
    mmio_write(0x8a8, 0x02040000);
    if (u1 == 1)
	set4160();

    setxferbase(0);
    extr(xdata, data, 28, 29);
    xfer(data << 4, 0x200, 3, 0x400, xdata, 1, 1, 2);

    ctxbase = (((s4.unk410.a & 0xfffff000) >> 8)&0x00ffffff) | (s4.unk410.b << 0x18);
    if (ctxbase != curctxbase) {
	curctxbase = ctxbase;
	mmwr(0x41a500, ctxbase, 0, 0);
	mmwr(0x41a504, 0x13, 0, 0);
    }

    setxferbase(curctxbase);
    wait4170();
    funk598(8);
    set4170_0(0);
    d76(cmd, data, 2);
    load5(1);
    load3(2);
    unk190 = s5.unk53c & 7;
    if (unk190 == 1) {
	uint32_t t, xt;

	mmio_write(0x86c, 0);
	mmwr(0x408a14, 0, 0, 0);
	mmwr(0x41a86c, 0, 0, 0);

	t = mmrd(0x400208, 0);
	// wtf?
	mmio_write(FUC_NEWSCRATCH7, t & 0xf);
	extr(xt, t, 4, 7);
	mmio_write(FUC_NEWSCRATCH7, xt);

	mmwr(0x400208, 0x80000004, 0, 0);
	for (i = 0; i != 8; i++) {
	    uint32_t mval, xval;
	    uint32_t val = s5.unk540[i];
	    extr(xval, val, 0, 2);
	    mval = xval;
	    extr(xval, val, 4, 6);
	    mval |= xval << 3;
	    extr(xval, val, 8, 10);
	    mval |= xval << 6;
	    extr(xval, val, 12, 14);
	    mval |= xval << 9;
	    extr(xval, val, 16, 18);
	    mval |= xval << 0xc;
	    extr(xval, val, 20, 22);
	    mval |= xval << 0xf;
	    extr(xval, val, 24, 26);
	    mval |= xval << 0x12;
	    extr(xval, val, 28, 30);
	    mval |= xval << 0x15;
	    mmwr(0x400204, mval, 0, 0);
	    mmwr(0x400200, 0x848 + i, 0, 0);
	}
	while (mmrd(0x400700, 0) & 0x0303fb7c);

	mmwr(0x400208, 0, 0, 0);

	mmio_write(0x86c, 0x10);
	mmwr(0x408a14, 0x10, 0, 0);
	mmwr(0x41a86c, 0x10, 0, 0);
    }

    tm = s5.unk51c & 7;
    if (tm != curs3pres) {
	curs3pres = tm;
	mmwr(0x41a500, tm, 0, 0);
	mmwr(0x41a504, 0x19, 0, 0);
    }

    if (curs3pres == 2 && curs3base != s5.unk520) {
	curs3base = s5.unk520;
	mmwr(0x41a500, s5.unk520, 0, 0);
	mmwr(0x41a504, 0x17, 0, 0);
    }

    tm = s5.unk528 & 7;
    if (tm != curs2pres) {
	curs2pres = tm;
	mmwr(0x41a500, tm, 0, 0);
	mmwr(0x41a504, 0x28, 0, 0);
    }

    if (curs2pres == 1 && curs2base != s5.unk52c) {
	curs2base = s5.unk52c;
	mmwr(0x41a500, s5.unk52c, 0, 0);
	mmwr(0x41a504, 0x26, 0, 0);
    }

    if (s5.marker != 0x600dc0de) {
	error(1); // inline
	clear4160();
	mmio_write(FUC_NEWSCRATCH0, 2);
	mmio_write(0x484, 0);
	mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80000);
	return 0;
    }

    mmio_write(FUC_NEWSCRATCH5, 0);
    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a800, 0, 0, 0);
    mmwr(0x41a500, data, 0, 0);
    mmwr(0x41a504, 2, 0, 0);

    if (s3.marker != 0xad0becab) {
	error(1); // inline
	mmio_write(FUC_NEWSCRATCH0, 2);
	clear4160();
	mmio_write(0x484, 0);
	mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80000);
	return 0;
    }

    funk312(unk18);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xc);
    waitstr();
    mmio_write_i(0x91c, 0x3f, 0);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 4);
    mmio_write(FUC_MMCTX_LOAD_COUNT, 0xcb);
    mmio_write(FUC_MMCTX_LOAD_SWBASE, s3.mmbase & 0xffff);

    mmctx(2, 1, 2, mmio_table1, nr_mmio_table1);
    if (curs2pres == 1) {
	setxferbase(curs2base);
	mmio_write(FUC_MMCTX_LOAD_COUNT, 0x1be);
	mmio_write(FUC_MMCTX_LOAD_SWBASE, s3.unk3a4[8].b & 0xffff);
	mmctx(2, 2, 2, mmio_table2, nr_mmio_table2);
	mmio_write(FUC_MMCTX_LOAD_COUNT, ropcount * 0x68);
	mmio_write(FUC_MMCTX_LOAD_SWBASE, s3.unk3ec & 0xffff); 
	mmctx(2, 9, 1, mmio_table3, nr_mmio_table3);
	mmctx(2, 0xa, 4, mmio_table4, nr_mmio_table4);
	setxferbase(curctxbase);
    }

    waitstr();
    (s5.unk5f8)++;

    while (!(mmio_read(FUC_DONE)&0x100));

    if (mmio_read(FUC_NEWSCRATCH5)) {
	mmio_write(0x424, 0);
	mmio_write(FUC_NEWSCRATCH0, 2);
	mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80000);
	return 0;
    }

    set4170_2(0);
    mmio_write(0x86c, 0);
    mmwr(0x408a14, 0, 0, 0);
    mmwr(0x41a86c, 0, 0, 0);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xd);
    waitstr();
    wait4170();
    set4170_0(0);

    mmio_write(0x874, data & 0x3fffffff);
    mmio_write(0x878, 4);
    mmio_write(0x8a8, 0x7000000);
    mmcnt = s5.unk510;
    if (mmcnt) {
	int scnt = ((mmcnt << 3) + 0xff) & 0xffffff00;
	// XXX what the fuck? hmm, could be artifact of someone using / operator on signed numbers.
	if (scnt < 0)
	    scnt += 0xff;

	setxferbase(((s5.unk514 >> 8)&0x00ffffff )  | (s5.unk518 << 0x18));
	scnt >>= 8;

	for (i = 0, j = 0; i < scnt; i++) {
	    xfer(0, i * 0x100, 6, 0x400, 0, 2, 1, 2);
	    struct pair *ptr = (struct pair *)(0x400);
	    //struct pair *ptr = (struct pair *)&s4;
	    do {
		mmwr(ptr->a, ptr->b, 0, 0);
		ptr++;
		if (++j == mmcnt)
		    goto out;
	    } while ((uint32_t)ptr != 0x500);
	}
out:
	setxferbase(curctxbase);
	s5.unk510 = 0;
	store5(2);
	mmio_write(0x874, data & 0x3fffffff);
	mmio_write(0x878, 0xd);
	mmio_write(0x8a8, 0x7000000);
    }

    clear4160();
    ach = 1;
    ach_chan = data;
    ach_ctxbase = curctxbase;
    ach_s5 = 1;
    mmio_write(0x430, 0);
    mmio_write(0x424, 0);
    mmio_write(FUC_NEWSCRATCH0, 4);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x80000);
#endif
    return 0;
}

void savectx (uint32_t cmd, uint32_t data, uint32_t u1, uint32_t u2)
{
    int i;
    int ctxbase, tm;
    uint32_t xdata;
	
    mmio_write(FUC_NEWSCRATCH_SET2, 0x40000);
    funk1f47(!u2);

    mmio_write(0x86c, 0x10);
    mmwr(0x408a14, 0x10, 0, 0);
    mmwr(0x41a86c, 0x10, 0, 0);
    funk312(unk18);

    if (u2) {
	mmio_write(0x874, data & 0x3fffffff);
	mmio_write(0x878, 8);
	mmio_write(0x8a8, 0x05020000);
	u1 = 0;
	set4160();
    } else {
	if (u1 == 2) {
	    mmio_write(0x874, data & 0x3fffffff);
	    mmio_write(0x878, 9);
	    mmio_write(0x8a8, 0x04030000);
	} else {
	    mmio_write(0x874, data & 0x3fffffff);
	    mmio_write(0x878, 0xa);
	    mmio_write(0x8a8, 0x05020000);
	}
	set4160();
	if (ach && data == ach_chan) {
	    setxferbase(curctxbase);
	    if (ach_s5)
		goto out2;
	    goto out1;
	}
    }


#if 1
    setxferbase(0);
    extr(xdata, data, 28, 29);
    xfer(data << 4, 0x200, 3, 0x400, xdata, 1, 1, 2);

    ctxbase = (((s4.unk410.a & 0xfffff000) >> 8)&0x00ffffff) | (s4.unk410.b << 0x18);
    if (ctxbase != curctxbase) {
	curctxbase = ctxbase;
	setxferbase(ctxbase);
	mmwr(0x41a500, curctxbase, 0, 0);
	mmwr(0x41a504, 0x13, 0, 0);
    }

    ach = 0;
    ach_chan = 0;
    ach_s5 = 0;
    setxferbase(curctxbase);
out1:
    load5(2);
out2:
    unk190 = s5.unk53c & 7;
    if (unk190 == 1) {
	for (i = 0; i < 8; i++)
	    s5.unk540[i] = mmrd(0x504730 + i * 4, 0);
    }
    tm = s5.unk51c & 7;
    if (curs3pres != tm) {
	curs3pres = tm;
	mmwr(0x41a500, tm, 0, 0);
	mmwr(0x41a504, 0x19, 0, 0);
    }

    if (curs3pres == 2 && curs3base != s5.unk520) {
	mmwr(0x41a500, s5.unk520, 0, 0);
	mmwr(0x41a504, 0x17, 0, 0);
    }

    if (tm && u2) {
	s5.unk520 = 0;
	s5.unk51c = 0;
    }

    tm = s5.unk528 & 7;
    if (curs2pres != tm) {
	curs2pres = tm;
	mmwr(0x41a500, tm, 0, 0);
	mmwr(0x41a504, 0x28, 0, 0);
    }
    if (curs2pres == 1 && curs2base != s5.unk52c) {
	curs2base = s5.unk52c;
	mmwr(0x41a500, s5.unk52c, 0, 0);
	mmwr(0x41a504, 0x26, 0, 0);
    }

    mmio_write(FUC_BAR_STATUS, 0);
    mmwr(0x41a800, 0, 0, 0);
    mmwr(0x41a500, data, 0, 0);
    if (u1 == 2) {
	mmwr(0x41a504, 1, 0, 0);
    } else {
	mmwr(0x41a504, 8, 0, 0);
    }

    funk312(unk18);
    mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xc);
    waitstr();
    //funk9e(0); // inlined
     mmio_write_i(0x91c, 0x3f, 0);
     mmio_write_i(FUC_STRAND_CMD, 0x3f, 3);

    mmio_write(FUC_MMCTX_SAVE_SWBASE, s3.mmbase & 0xffff);
    mmctx(1, 1, 2, mmio_table1, nr_mmio_table1);
    if (curs2pres == 1) {
	waitstr();
	setxferbase(curs2base);

	mmio_write(FUC_MMCTX_SAVE_SWBASE, s3.unk3a4[8].b & 0xffff); 
	mmctx(1, 2, 2, mmio_table2, nr_mmio_table2);
	mmio_write(FUC_MMCTX_SAVE_SWBASE, s3.unk3ec & 0xffff);
	mmctx(1, 9, 1, mmio_table3, nr_mmio_table3);
	mmctx(1, 0xa, 4, mmio_table4, nr_mmio_table4);
	setxferbase(curctxbase);
    }

    waitstr();
    s5.unk5f4 += 1;
    s3.marker = 0xad0becab;
    store3(1);
    store5(2);

    /*fix this*/
    //while (!(mmio_read(FUC_DONE)&0x100));
    while ((mmio_read(FUC_DONE)&0x100));
    
    funk312(unk18);
    waitdone_12();
    mmio_write(0xa10, 5);
    while (mmio_read(0xa10));

    if (!u1) {
	set4170_2(0);
	mmio_write(0x86c, 0);
	mmwr(0x408a14, 0, 0, 0);
	mmwr(0x41a86c, 0, 0, 0);
	mmio_write_i(FUC_STRAND_CMD, 0x3f, 0xd);
	waitstr();
	clear4160();
	wait4170();
	set4170_0(0);
	mmio_write(0x874, data & 0x3fffffff);
	mmio_write(0x878, 4);
	mmio_write(0x8a8, 0x2000000);
    }

    mmio_write(0x430, 0);
    // ack savectx success
#endif
    mmio_write(FUC_NEWSCRATCH0, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x40000);

}

void unkc9(uint32_t cmd, uint32_t data)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 0x1000);
    mmio_write(0xb08, 0);

    savectx(cmd, data, 0, 1);

    while (qget != qput) {
	cmd = queue[qget & 7].cmd & ~0x7800;
	if (++qget == 0x10)
	    qget = 0;

	if (cmd != 0x10000) {
	    mmio_write(FUC_NEWSCRATCH_SET0, 2);
	    break;
	}

	mmio_write(0xb0c, 1);
    }

    mmio_write(0xb08, 1);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x1000);
}

void autochansw(void)
{
    uint32_t old, new;

    mmio_write(FUC_NEWSCRATCH_SET2, 0x10000);
    old = mmio_read(0xb00);
    new = mmio_read(0xb04);
#if 0
    if (old & 0x80000000) { //fix this
	uint32_t newm = new & 0x3fffffff;
	uint32_t oldm = old & 0x3fffffff;
	if (new & 0x80000000) { //fix this
	    if (newm != oldm) {
		if (!fakeautosw)
		    savectx(0, old, 2, 0);
		mmio_write(0xb00, new);
		if (!fakeautosw) {
		    loadctx(0, new, 2);
		}
	    }
	} else {
	    if (newm == oldm) {
		if (!fakeautosw)
		    savectx(0, old, 0, 0);

		mmio_write(0xb00, new);
	    }
	}
    } else {
	if (new & 0x80000000) { //fix this
	    mmio_write(0xb00, new);
	    if (!fakeautosw) {
		loadctx(0, new, 1);
	    }
	}
    }
#endif
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x10000);
}

void unkca(uint32_t cmd, uint32_t data)
{
    mmio_write(FUC_NEWSCRATCH_SET2, 0x20000);
    savectx(0, mmio_read(0xb00), 2, 0);
    mmio_write(0xb00, data);
    loadctx(0, data, 2);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 0x20000);
}


void work(void)
{
    uint32_t cmd, data;
    mmio_write(FUC_NEWSCRATCH_SET2, 8);
    if (q2get != q2put) {
	struct qe *ptr = &q2[q2get & 7];
	cmd = ptr->cmd & ~0x7800;
	data = ptr->data;
	if (++q2get == 0x10)
	    q2get = 0;
    } else if (qget != qput) {
	struct qe *ptr = &queue[qget & 7];
	cmd = ptr->cmd & ~0x7800;
	data = ptr->data;
	if (++qget == 0x10)
	    qget = 0;
    } else {
	mmio_write(FUC_NEWSCRATCH_CLEAR2, 8);
	return;
    }


    switch (cmd) {
	case 1:
	    savectx(cmd, data, 2, 0);
	    break;
	case 2:
	    loadctx(cmd, data, 1);
	    break;
	case 8:
	    savectx(cmd, data, 0, 0);
	    break;
	case 0x21:
	    unk18 = data;
	    mmwr(0x41a500, data, 0, 0);
	    mmwr(0x41a504, cmd, 0, 0);
	    break;
	case 0xb:
	    unkcb(cmd, data);
	    break;
	case 9:
	    unkc9(cmd, data);
	    break;
	case 0xa:
	    unkca(cmd, data);
	    break;
	case 3:
	    d76(cmd, data, 1);
	    mmio_write(FUC_NEWSCRATCH0, 0x10);
	    break;
	case 4:
	    unkc4(cmd, data);
	    break;
	case 5:
	    unkc5(cmd, data);
	    break;
	case 6:
	    unkc6(cmd, data);
	    break;
	case 0x10:
	    unkc10(cmd, data);
	    break;
	case 0x16:
	    unkc16(cmd, data);
	    break;
	case 0x25:
	    unkc25(cmd, data);
	    break;
	case 0x11:
	    unkc11(cmd, data);
	    break;
	case 0x29:
	    //selftest(cmd, data);
	    break;
	case 0x15:
	    unkc15(cmd, data);
	    break;
	case 0x12:
	    done = 1;
	    break;
	case 0xc:
	    setfakeautosw(cmd, data); // inline
	    break;
	case 0x555:
	 //  autochansw();
	//   mmio_write(0xb0c, 1);
	    break;
	case 0x10000:
	    autochansw();
	    mmio_write(0xb0c, 1);
	    break;
	case 0x10002:
	    fwmthd(cmd, data);
	    break;
	case 0:
	case 0x13:
	case 0x14:
	case 0x17:
	case 0x18:
	case 0x19:
	case 0x26:
	case 0x27:
	case 0x28:
	case 0x10001:
	case 0x10003:
	    // nop
	    break;
	default:
	    error(0x11); // inline
	    break;
    }

    mmio_write(FUC_NEWSCRATCH1, 2);
    mmio_write(FUC_NEWSCRATCH_CLEAR2, 8);
}

/*
 * entry point into the microcontroller code.
 */
int main(void)
{
    int i;
    done = 0;
    mmio_write(FUC_NEWSCRATCH_SET2,1);
    mmio_write(FUC_ACCESS_EN, FUC_AE_BIT_FIFO);

    // interrupt handler is already set by hardcoding.
    // ih_setup(ih);


    mmio_write(FUC_INTR_DISPATCH, 0); // fuc interrupt 0
    mmio_write(FUC_INTR_EN_SET, 0x8704);
    mmio_write(0x00c, 4);

    mmio_write_i(FUC_MMCTX_INTR_ROUTE, 0, 0x2003);
    mmio_write_i(FUC_MMCTX_INTR_ROUTE, 1, 0x2004);
    mmio_write_i(FUC_MMCTX_INTR_ROUTE, 2, 0x200b);
    mmio_write_i(FUC_MMCTX_INTR_ROUTE, 7, 0x200c);
    mmio_write_i(0xc14, 0x10, 0x202d);

    set_flags(ie0);
    set_flags($p2);


    mmio_write(FUC_MMCTX_SAVE_SWBASE, 0);
    mmio_write(FUC_MMCTX_LOAD_SWBASE, 0);

    tpcount = mmrd(HUB_BASE + FUC_UNITS, 0) & 0x1f;
    ropcount = (mmrd(HUB_BASE + FUC_UNITS, 0) >> 0x10) & 0x1f;
    mmio_write(FUC_BAR_REQMASK, (1 << tpcount) - 1);
    mmio_write(0x410, (1 << tpcount) - 1); 
    mmio_write(0xc24, 0xffffffff);//INTR_UP_ENABLE??

    for (i = 0; i < tpcount; i++)
	while (mmrd(GPC_BASE+ 0x800 + i * GPC_OFFSET, 0) != 0x87654321);

    mmio_write(FUC_NEWSCRATCH0, 1); // this acks the driver

    while (1) {
	sleep($p2);
	work();
	if (done)
	    break;
	set_flags($p2);
	if (qget != qput)
	    clr_flags($p2);
    }

    mmio_write(FUC_NEWSCRATCH_CLEAR2, 1);

    return 0;
}
