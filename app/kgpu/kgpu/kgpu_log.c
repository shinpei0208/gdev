/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Log functions used by both kernel and user space.
 */
#include "kgpu.h"

#ifndef __KERNEL__

#include <stdio.h>
#include <stdarg.h>

#define printk printf
#define vprintk vprintf

#else

#include <linux/kernel.h>
#include <linux/module.h>

#endif /* __KERNEL__ */

#ifdef __KGPU_LOG_LEVEL__
int kgpu_log_level = __KGPU_LOG_LEVEL__;
#else
int kgpu_log_level = KGPU_LOG_ALERT;
#endif

void
kgpu_generic_log(int level, const char *module, const char *filename,
		 int lineno, const char *func, const char *fmt, ...)
{
    va_list args;
    
    if (level < kgpu_log_level)
	return;
    
    switch(level) {
    case KGPU_LOG_INFO:
	printk("[%s] %s::%d %s() INFO: ", module, filename, lineno, func);
	break;
    case KGPU_LOG_DEBUG:
	printk("[%s] %s::%d %s() DEBUG: ", module, filename, lineno, func);
	break;
    case KGPU_LOG_ALERT:
	printk("[%s] %s::%d %s() ALERT: ", module, filename, lineno, func);
	break;
    case KGPU_LOG_ERROR:
	printk("[%s] %s::%d %s() ERROR: ", module, filename, lineno, func);
	break;
    case KGPU_LOG_PRINT:
	printk("[%s] %s::%d %s(): ", module, filename, lineno, func);
	break;
    default:
	break;
    }
    
    va_start(args, fmt);	
    vprintk(fmt, args);
    va_end(args);
}

#ifdef __KERNEL__

EXPORT_SYMBOL_GPL(kgpu_generic_log);
EXPORT_SYMBOL_GPL(kgpu_log_level);

#endif /* __KERNEL__ */
