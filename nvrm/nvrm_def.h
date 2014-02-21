/*
 * Copyright (C) 2013 Marcin Kościelnicki <koriakin@0x04.net>
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
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef NVRM_DEF_H
#define NVRM_DEF_H

#define NVRM_CHECK_VERSION_STR_CMD_STRICT		0
#define NVRM_CHECK_VERSION_STR_CMD_RELAXED		'1'
#define NVRM_CHECK_VERSION_STR_CMD_OVERRIDE		'2'
#define NVRM_CHECK_VERSION_STR_REPLY_UNRECOGNIZED	0
#define NVRM_CHECK_VERSION_STR_REPLY_RECOGNIZED		1

#define NVRM_FIFO_ENG_GRAPH	1
#define NVRM_FIFO_ENG_COPY0	2
#define NVRM_FIFO_ENG_COPY1	3
#define NVRM_FIFO_ENG_COPY2	4
#define NVRM_FIFO_ENG_VP	5
#define NVRM_FIFO_ENG_ME	6
#define NVRM_FIFO_ENG_PPP	7
#define NVRM_FIFO_ENG_BSP	8
#define NVRM_FIFO_ENG_MPEG	9
#define NVRM_FIFO_ENG_SOFTWARE	10
#define NVRM_FIFO_ENG_CRYPT	11
#define NVRM_FIFO_ENG_VCOMP	12
#define NVRM_FIFO_ENG_VENC	13
/* XXX 15? */
/* XXX 14? apparently considered to be valid by blob logic */
#define NVRM_FIFO_ENG_UNK16	16

#define NVRM_COMPUTE_MODE_DEFAULT		0
#define NVRM_COMPUTE_MODE_EXCLUSIVE_THREAD	1
#define NVRM_COMPUTE_MODE_PROHIBITED		2
#define NVRM_COMPUTE_MODE_EXCLUSIVE_PROCESS	3

#define NVRM_PERSISTENCE_MODE_ENABLED	0
#define NVRM_PERSISTENCE_MODE_DISABLED	1

#endif
