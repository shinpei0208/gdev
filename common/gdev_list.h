/*
 * Copyright 2011 Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
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

#ifndef __GDEV_LIST_H__
#define __GDEV_LIST_H__

#ifndef NULL
#define NULL 0
#endif

/* a list structure: we could use Linux's list_head, but it's not available
   in user-space - hence use our own list structure. */
struct gdev_list {
    struct gdev_list *next;
    struct gdev_list *prev;
	void *container;
};

static inline void gdev_list_init(struct gdev_list *entry, void *container)
{
	entry->next = entry->prev = NULL;
	entry->container = container;
}

static inline void gdev_list_add
(struct gdev_list *entry, struct gdev_list *head)
{
	struct gdev_list *next = head->next;

	entry->next = next;
	if (next)
		next->prev = entry;
	entry->prev = head; /* link to the head. */
	head->next = entry;
}

static inline void gdev_list_del(struct gdev_list *entry)
{
	struct gdev_list *next = entry->next;
	struct gdev_list *prev = entry->prev;

	/* if prev is null, @entry points to the head, hence something wrong. */
	if (prev) {
		prev->next = next;
		if (next)
			next->prev = prev;
		entry->next = entry->prev = NULL;
	}
}

static inline int gdev_list_empty(struct gdev_list *entry)
{
	return entry->next == entry->prev;
}

static inline struct gdev_list *gdev_list_head(struct gdev_list *head)
{
	return head ? head->next : NULL;
}

static inline void *gdev_list_container(struct gdev_list *entry)
{
	return entry ? entry->container : NULL;
}

#define gdev_list_for_each(p, list, entry)					\
	for (p = gdev_list_container(gdev_list_head(list));		\
		 p != NULL;											\
		 p = gdev_list_container((p)->entry.next))

#define gdev_list_add_ordered(entry, head, member)	\
	do {										\
		struct gdev_list *p, *tail = head;		\
		(entry)->next = (entry)->prev = NULL;	\
		gdev_list_for_each(p, head) {			\
			if ((entry)->member <= p->member) {	\
				gdev_list_add(entry, p);		\
				break;							\
			}									\
			tail = p;							\
		}										\
		if (gdev_list_empty(entry))				\
			gdev_list_add(entry, tail);			\
	} while (0)

#endif
