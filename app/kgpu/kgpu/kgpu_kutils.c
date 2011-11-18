/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU kernel module utilities
 *
 */
#include "kkgpu.h"
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/current.h>

struct page* kgpu_v2page(unsigned long v)
{
    struct page *p = NULL;
    pgd_t *pgd = pgd_offset(current->mm, v);

    if (!pgd_none(*pgd)) {
	pud_t *pud = pud_offset(pgd, v);
	if (!pud_none(*pud)) {
	    pmd_t *pmd = pmd_offset(pud, v);
	    if (!pmd_none(*pmd)) {
		pte_t *pte;

		pte = pte_offset_map(pmd, v);
		if (pte_present(*pte))
		    p = pte_page(*pte);
		
		/*
		 * although KGPU doesn't support x86_32, but in case
		 * some day it does, the pte_unmap should not be called
		 * because we want the pte stay in mem.
		 */
		pte_unmap(pte);
	    }
	}
    }
    if (!p)
	kgpu_log(KGPU_LOG_ALERT, "bad address 0x%lX\n", v);
    return p;
}

