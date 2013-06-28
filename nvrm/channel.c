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

#include "nvrm_priv.h"
#include "nvrm_class.h"
#include "nvrm_mthd.h"
#include <stdlib.h>
#include <sys/mman.h>

struct nvrm_channel *nvrm_channel_create_ib(struct nvrm_vspace *vas, uint32_t cls, struct nvrm_bo *ib) {
	struct nvrm_channel *chan = calloc(sizeof *chan, 1);
	if (!chan)
		goto out_alloc;
	chan->ctx = vas->ctx;
	chan->dev = vas->dev;
	chan->vas = vas;
	chan->cls = cls;
	chan->oerr = nvrm_handle_alloc(chan->ctx);
	chan->oedma = nvrm_handle_alloc(chan->ctx);
	chan->ofifo = nvrm_handle_alloc(chan->ctx);

	if (nvrm_ioctl_memory(chan->ctx, chan->dev->odev, chan->dev->odev, chan->oerr, 0xd001, 0x3a000000, 0, 0x1000))
		goto out_err;

	if (nvrm_ioctl_create_dma(chan->ctx, chan->oerr, chan->oedma, NVRM_CLASS_DMA_READ, 0x20100000, 0, 0xfff))
		goto out_edma;

	struct nvrm_create_fifo_ib arg = {
		.error_notify = chan->oedma,
		.dma = chan->vas->odma,
		.ib_addr = ib->gpu_addr,
		.ib_entries = ib->size / 8,
	};
	if (nvrm_ioctl_create(chan->ctx, chan->dev->odev, chan->ofifo, cls, &arg))
		goto out_fifo;

	if (nvrm_ioctl_host_map(chan->ctx, chan->dev->osubdev, chan->ofifo, 0, 0x200, &chan->fifo_foffset))
		goto out_fifo_map;
	chan->fifo_mmap = mmap(0, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, chan->dev->fd, chan->fifo_foffset & ~0xfff);
	if (chan->fifo_mmap == MAP_FAILED)
		goto out_mmap;

	return chan;

out_mmap:
	nvrm_ioctl_host_unmap(chan->ctx, chan->dev->osubdev, chan->ofifo, chan->fifo_foffset);
out_fifo_map:
	nvrm_ioctl_destroy(chan->ctx, chan->dev->odev, chan->ofifo);
out_fifo:
	nvrm_ioctl_destroy(chan->ctx, chan->oerr, chan->oedma);
out_edma:
	nvrm_ioctl_destroy(chan->ctx, chan->dev->odev, chan->oerr);
out_err:
	nvrm_handle_free(chan->ctx, chan->oerr);
	nvrm_handle_free(chan->ctx, chan->oedma);
	nvrm_handle_free(chan->ctx, chan->ofifo);
out_alloc:
	return 0;
}

int nvrm_channel_activate(struct nvrm_channel *chan) {
	if (chan->cls >= 0xa06f) {
		struct nvrm_mthd_fifo_ib_activate arg = {
			1,
		};
		return nvrm_ioctl_call(chan->ctx, chan->ofifo, NVRM_MTHD_FIFO_IB_ACTIVATE, &arg, sizeof arg);
	}
	return 0;
}

void nvrm_channel_destroy(struct nvrm_channel *chan) {
	while (chan->echain) {
		struct nvrm_eng *eng = chan->echain;
		chan->echain = eng->next;
		nvrm_ioctl_destroy(chan->ctx, chan->ofifo, eng->handle);
		nvrm_handle_free(chan->ctx, eng->handle);
		free(eng);
	}
	munmap(chan->fifo_mmap, 0x1000);
	nvrm_ioctl_host_unmap(chan->ctx, chan->dev->osubdev, chan->ofifo, chan->fifo_foffset);
	nvrm_ioctl_destroy(chan->ctx, chan->dev->odev, chan->ofifo);
	nvrm_ioctl_destroy(chan->ctx, chan->oerr, chan->oedma);
	nvrm_ioctl_destroy(chan->ctx, chan->dev->odev, chan->oerr);
	nvrm_handle_free(chan->ctx, chan->oerr);
	nvrm_handle_free(chan->ctx, chan->oedma);
	nvrm_handle_free(chan->ctx, chan->ofifo);
}

void *nvrm_channel_host_map_regs(struct nvrm_channel *chan) {
	return chan->fifo_mmap + (chan->fifo_foffset & 0xfff);
}

void *nvrm_channel_host_map_errnot(struct nvrm_channel *chan);

struct nvrm_eng *nvrm_eng_create(struct nvrm_channel *chan, uint32_t eid, uint32_t cls) {
	struct nvrm_eng *eng = calloc(sizeof *eng, 1);
	if (!eng)
		goto out_alloc;
	eng->chan = chan;
	eng->handle = nvrm_handle_alloc(chan->ctx);
	if (nvrm_ioctl_create(chan->ctx, chan->ofifo, eng->handle, cls, 0))
		goto out_eng;
	eng->next = chan->echain;
	chan->echain = eng;
	return eng;

out_eng:
	nvrm_handle_free(chan->ctx, eng->handle);
	free(eng);
out_alloc:
	return 0;
}
