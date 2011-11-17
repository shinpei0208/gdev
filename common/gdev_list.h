/*
* Copyright 2011 Shinpei Kato
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
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
typedef struct gdev_list {
    struct gdev_list *next;
    struct gdev_list *prev;
	void *container;
} gdev_list_t;

static inline void __gdev_list_init(struct gdev_list *entry, void *container)
{
	entry->next = entry->prev = NULL;
	entry->container = container;
}

static inline void __gdev_list_add
(struct gdev_list *entry, struct gdev_list *head)
{
	struct gdev_list *next = head->next;
	
	entry->next = next;
	if (next)
		next->prev = entry;
	entry->prev = NULL; /* don't link to the head. */
	head->next = entry;
}

static inline void __gdev_list_del(struct gdev_list *entry)
{
	struct gdev_list *next = entry->next;
	struct gdev_list *prev = entry->prev;
	
	if (next) {
		next->prev = entry->prev;
	}
	if (prev) {
		prev->next = entry->next;
	}
	entry->next = entry->prev = NULL;
}

static inline struct gdev_list *__gdev_list_head(struct gdev_list *head)
{
	if (!head)
		return NULL;
	return head->next;
}

static inline void *__gdev_list_container(struct gdev_list *entry)
{
	if (!entry)
		return NULL;
	return entry->container;
}

#define gdev_list_for_each(p, list)							\
	for (p = __gdev_list_container(__gdev_list_head(list));	\
		 p != NULL;											\
		 p = __gdev_list_container((p)->list_entry.next))

#endif
