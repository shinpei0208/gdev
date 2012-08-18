#ifndef __ASMINSN_H__
#define __ASMINSN_H__

/* sleep until interrupt */
#define sleep(b) __asm__ volatile("sleep "#b)
/* set bit in the $flags register */
#define set_flags(b) __asm__ volatile("bset\t$flags "#b)
/* clear bit in the $flags register */
#define clr_flags(b) __asm__ volatile("bclr\t$flags "#b)

#define set_xdbase(x) __asm__ volatile("mov\t$xdbase %0" :: "r"(x))
#define set_xtargets(x) __asm__ volatile("mov\t$xtargets %0" :: "r"(x))
    
#define xdld(offset, size_addr) \
    __asm__ volatile("xdld\t%0 %1" :: "r"(offset), "r"(size_addr))

#define xdst(offset, size_addr)	\
    __asm__ volatile("xdst\t%0 %1" :: "r"(offset), "r"(size_addr))

#define xdwait() \
    __asm__ volatile("xdwait");

#define extrs(dst, src, a, b)					\
    __asm__ volatile("extrs\t%0 %1 "#a":"#b : "=r"(dst): "r"(x))

#define extr(dst, src, a, b)					\
    __asm__ volatile("extr\t%0 %1 "#a":"#b : "=r"(dst): "r"(src))

#endif
