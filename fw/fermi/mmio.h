#ifndef __FUC_IO_H__
#define __FUC_IO_H__

#include "types.h"


static inline int sar(int val, int shcnt)
{
    int x;
    __asm__ volatile ("sar\tb32\t%0 %1 %2" :"=r"(x) : "r"(val), "r"(shcnt));
    return x;
}



static inline void mmio_write(int addr, int val)
{
    __asm__ volatile ("iowr\tI[%0] %1" :: "r"(addr << 6), "r"(val));
}

static inline int mmio_read(int addr)
{
    int x;
    __asm__ volatile ("iord\t%0 I[%1]" :"=r"(x) : "r"(addr << 6));
    return x;
}


static inline void mmio_write_n(int addr, int val)
{
    __asm__ volatile ("iowr\tI[%0] %1" :: "r"(addr), "r"(val));
   // __asm__ volatile ("iowr\tI[%0] %1" :: "r"((addr + i * 4) << 6), "r"(val));
   // __asm__ volatile ("iowr\tI[%0] %1" :: "r"(addr << 6 + i * 4), "r"(val));
}

static inline void mmio_write_i(int addr, int i, int val)
{
    addr = (addr << 6) + i * 4;
    __asm__ volatile ("iowr\tI[%0] %1" :: "r"(addr), "r"(val));
   // __asm__ volatile ("iowr\tI[%0] %1" :: "r"((addr + i * 4) << 6), "r"(val));
   // __asm__ volatile ("iowr\tI[%0] %1" :: "r"(addr << 6 + i * 4), "r"(val));
}

static inline int mmio_read_i(int addr, int i)
{
    int x;
    addr = (addr<<6)+ i*4;
//    __asm__ volatile ("iord\t%0 I[%1]" :"=r"(x) : "r"(addr << 6 + i * 4));
//    __asm__ volatile ("iord\t%0 I[%1]" :"=r"(x) : "r"((addr + i * 4)  << 6));
    __asm__ volatile ("iord\t%0 I[%1]" :"=r"(x) : "r"(addr));
    return x;
}

#endif
