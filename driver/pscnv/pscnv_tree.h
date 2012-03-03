/*
 * Originally sys/tree.h from FreeBSD. Changes:
 *  - SPLAY removed
 *  - name changed to avoid collisions
 */

/*
 * Copyright 2002 Niels Provos <provos@citi.umich.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef	__PSCNV_TREE_H__
#define	__PSCNV_TREE_H__

#define __unused __attribute__((__unused__))

/* Macros that define a red-black tree */
#define PSCNV_RB_HEAD(name, type)						\
struct name {								\
	struct type *rbh_root; /* root of the tree */			\
}

#define PSCNV_RB_INITIALIZER(root)						\
	{ NULL }

#define PSCNV_RB_INIT(root) do {						\
	(root)->rbh_root = NULL;					\
} while (/*CONSTCOND*/ 0)

#define PSCNV_RB_BLACK	0
#define PSCNV_RB_RED		1
#define PSCNV_RB_ENTRY(type)							\
struct {								\
	struct type *rbe_left;		/* left element */		\
	struct type *rbe_right;		/* right element */		\
	struct type *rbe_parent;	/* parent element */		\
	int rbe_color;			/* node color */		\
}

#define PSCNV_RB_LEFT(elm, field)		(elm)->field.rbe_left
#define PSCNV_RB_RIGHT(elm, field)		(elm)->field.rbe_right
#define PSCNV_RB_PARENT(elm, field)		(elm)->field.rbe_parent
#define PSCNV_RB_COLOR(elm, field)		(elm)->field.rbe_color
#define PSCNV_RB_ROOT(head)			(head)->rbh_root
#define PSCNV_RB_EMPTY(head)			(PSCNV_RB_ROOT(head) == NULL)

#define PSCNV_RB_SET(elm, parent, field) do {					\
	PSCNV_RB_PARENT(elm, field) = parent;					\
	PSCNV_RB_LEFT(elm, field) = PSCNV_RB_RIGHT(elm, field) = NULL;		\
	PSCNV_RB_COLOR(elm, field) = PSCNV_RB_RED;					\
} while (/*CONSTCOND*/ 0)

#define PSCNV_RB_SET_BLACKRED(black, red, field) do {				\
	PSCNV_RB_COLOR(black, field) = PSCNV_RB_BLACK;				\
	PSCNV_RB_COLOR(red, field) = PSCNV_RB_RED;					\
} while (/*CONSTCOND*/ 0)

#ifndef PSCNV_RB_AUGMENT
#define PSCNV_RB_AUGMENT(x) (void)(x)
#endif

#define PSCNV_RB_ROTATE_LEFT(head, elm, tmp, field) do {			\
	(tmp) = PSCNV_RB_RIGHT(elm, field);					\
	if ((PSCNV_RB_RIGHT(elm, field) = PSCNV_RB_LEFT(tmp, field)) != NULL) {	\
		PSCNV_RB_PARENT(PSCNV_RB_LEFT(tmp, field), field) = (elm);		\
	}								\
	PSCNV_RB_AUGMENT(elm);						\
	if ((PSCNV_RB_PARENT(tmp, field) = PSCNV_RB_PARENT(elm, field)) != NULL) {	\
		if ((elm) == PSCNV_RB_LEFT(PSCNV_RB_PARENT(elm, field), field))	\
			PSCNV_RB_LEFT(PSCNV_RB_PARENT(elm, field), field) = (tmp);	\
		else							\
			PSCNV_RB_RIGHT(PSCNV_RB_PARENT(elm, field), field) = (tmp);	\
	} else								\
		(head)->rbh_root = (tmp);				\
	PSCNV_RB_LEFT(tmp, field) = (elm);					\
	PSCNV_RB_PARENT(elm, field) = (tmp);					\
	PSCNV_RB_AUGMENT(tmp);						\
	if ((PSCNV_RB_PARENT(tmp, field)))					\
		PSCNV_RB_AUGMENT(PSCNV_RB_PARENT(tmp, field));			\
} while (/*CONSTCOND*/ 0)

#define PSCNV_RB_ROTATE_RIGHT(head, elm, tmp, field) do {			\
	(tmp) = PSCNV_RB_LEFT(elm, field);					\
	if ((PSCNV_RB_LEFT(elm, field) = PSCNV_RB_RIGHT(tmp, field)) != NULL) {	\
		PSCNV_RB_PARENT(PSCNV_RB_RIGHT(tmp, field), field) = (elm);		\
	}								\
	PSCNV_RB_AUGMENT(elm);						\
	if ((PSCNV_RB_PARENT(tmp, field) = PSCNV_RB_PARENT(elm, field)) != NULL) {	\
		if ((elm) == PSCNV_RB_LEFT(PSCNV_RB_PARENT(elm, field), field))	\
			PSCNV_RB_LEFT(PSCNV_RB_PARENT(elm, field), field) = (tmp);	\
		else							\
			PSCNV_RB_RIGHT(PSCNV_RB_PARENT(elm, field), field) = (tmp);	\
	} else								\
		(head)->rbh_root = (tmp);				\
	PSCNV_RB_RIGHT(tmp, field) = (elm);					\
	PSCNV_RB_PARENT(elm, field) = (tmp);					\
	PSCNV_RB_AUGMENT(tmp);						\
	if ((PSCNV_RB_PARENT(tmp, field)))					\
		PSCNV_RB_AUGMENT(PSCNV_RB_PARENT(tmp, field));			\
} while (/*CONSTCOND*/ 0)

/* Generates prototypes and inline functions */
#define PSCNV_RB_PROTOTYPE(name, type, field, cmp)				\
	PSCNV_RB_PROTOTYPE_INTERNAL(name, type, field, cmp,)
#define	PSCNV_RB_PROTOTYPE_STATIC(name, type, field, cmp)			\
	PSCNV_RB_PROTOTYPE_INTERNAL(name, type, field, cmp, __unused static)
#define PSCNV_RB_PROTOTYPE_INTERNAL(name, type, field, cmp, attr)		\
attr void name##_PSCNV_RB_INSERT_COLOR(struct name *, struct type *);		\
attr void name##_PSCNV_RB_REMOVE_COLOR(struct name *, struct type *, struct type *);\
attr struct type *name##_PSCNV_RB_REMOVE(struct name *, struct type *);	\
attr struct type *name##_PSCNV_RB_INSERT(struct name *, struct type *);	\
attr struct type *name##_PSCNV_RB_FIND(struct name *, struct type *);		\
attr struct type *name##_PSCNV_RB_NFIND(struct name *, struct type *);	\
attr struct type *name##_PSCNV_RB_NEXT(struct type *);			\
attr struct type *name##_PSCNV_RB_PREV(struct type *);			\
attr struct type *name##_PSCNV_RB_MINMAX(struct name *, int);			\
									\

/* Main rb operation.
 * Moves node close to the key of elm to top
 */
#define	PSCNV_RB_GENERATE(name, type, field, cmp)				\
	PSCNV_RB_GENERATE_INTERNAL(name, type, field, cmp,)
#define	PSCNV_RB_GENERATE_STATIC(name, type, field, cmp)			\
	PSCNV_RB_GENERATE_INTERNAL(name, type, field, cmp, __unused static)
#define PSCNV_RB_GENERATE_INTERNAL(name, type, field, cmp, attr)		\
attr void								\
name##_PSCNV_RB_INSERT_COLOR(struct name *head, struct type *elm)		\
{									\
	struct type *parent, *gparent, *tmp;				\
	while ((parent = PSCNV_RB_PARENT(elm, field)) != NULL &&		\
	    PSCNV_RB_COLOR(parent, field) == PSCNV_RB_RED) {			\
		gparent = PSCNV_RB_PARENT(parent, field);			\
		if (parent == PSCNV_RB_LEFT(gparent, field)) {		\
			tmp = PSCNV_RB_RIGHT(gparent, field);			\
			if (tmp && PSCNV_RB_COLOR(tmp, field) == PSCNV_RB_RED) {	\
				PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_BLACK;	\
				PSCNV_RB_SET_BLACKRED(parent, gparent, field);\
				elm = gparent;				\
				continue;				\
			}						\
			if (PSCNV_RB_RIGHT(parent, field) == elm) {		\
				PSCNV_RB_ROTATE_LEFT(head, parent, tmp, field);\
				tmp = parent;				\
				parent = elm;				\
				elm = tmp;				\
			}						\
			PSCNV_RB_SET_BLACKRED(parent, gparent, field);	\
			PSCNV_RB_ROTATE_RIGHT(head, gparent, tmp, field);	\
		} else {						\
			tmp = PSCNV_RB_LEFT(gparent, field);			\
			if (tmp && PSCNV_RB_COLOR(tmp, field) == PSCNV_RB_RED) {	\
				PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_BLACK;	\
				PSCNV_RB_SET_BLACKRED(parent, gparent, field);\
				elm = gparent;				\
				continue;				\
			}						\
			if (PSCNV_RB_LEFT(parent, field) == elm) {		\
				PSCNV_RB_ROTATE_RIGHT(head, parent, tmp, field);\
				tmp = parent;				\
				parent = elm;				\
				elm = tmp;				\
			}						\
			PSCNV_RB_SET_BLACKRED(parent, gparent, field);	\
			PSCNV_RB_ROTATE_LEFT(head, gparent, tmp, field);	\
		}							\
	}								\
	PSCNV_RB_COLOR(head->rbh_root, field) = PSCNV_RB_BLACK;			\
}									\
									\
attr void								\
name##_PSCNV_RB_REMOVE_COLOR(struct name *head, struct type *parent, struct type *elm) \
{									\
	struct type *tmp;						\
	while ((elm == NULL || PSCNV_RB_COLOR(elm, field) == PSCNV_RB_BLACK) &&	\
	    elm != PSCNV_RB_ROOT(head)) {					\
		if (PSCNV_RB_LEFT(parent, field) == elm) {			\
			tmp = PSCNV_RB_RIGHT(parent, field);			\
			if (PSCNV_RB_COLOR(tmp, field) == PSCNV_RB_RED) {		\
				PSCNV_RB_SET_BLACKRED(tmp, parent, field);	\
				PSCNV_RB_ROTATE_LEFT(head, parent, tmp, field);\
				tmp = PSCNV_RB_RIGHT(parent, field);		\
			}						\
			if ((PSCNV_RB_LEFT(tmp, field) == NULL ||		\
			    PSCNV_RB_COLOR(PSCNV_RB_LEFT(tmp, field), field) == PSCNV_RB_BLACK) &&\
			    (PSCNV_RB_RIGHT(tmp, field) == NULL ||		\
			    PSCNV_RB_COLOR(PSCNV_RB_RIGHT(tmp, field), field) == PSCNV_RB_BLACK)) {\
				PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_RED;		\
				elm = parent;				\
				parent = PSCNV_RB_PARENT(elm, field);		\
			} else {					\
				if (PSCNV_RB_RIGHT(tmp, field) == NULL ||	\
				    PSCNV_RB_COLOR(PSCNV_RB_RIGHT(tmp, field), field) == PSCNV_RB_BLACK) {\
					struct type *oleft;		\
					if ((oleft = PSCNV_RB_LEFT(tmp, field)) \
					    != NULL)			\
						PSCNV_RB_COLOR(oleft, field) = PSCNV_RB_BLACK;\
					PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_RED;	\
					PSCNV_RB_ROTATE_RIGHT(head, tmp, oleft, field);\
					tmp = PSCNV_RB_RIGHT(parent, field);	\
				}					\
				PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_COLOR(parent, field);\
				PSCNV_RB_COLOR(parent, field) = PSCNV_RB_BLACK;	\
				if (PSCNV_RB_RIGHT(tmp, field))		\
					PSCNV_RB_COLOR(PSCNV_RB_RIGHT(tmp, field), field) = PSCNV_RB_BLACK;\
				PSCNV_RB_ROTATE_LEFT(head, parent, tmp, field);\
				elm = PSCNV_RB_ROOT(head);			\
				break;					\
			}						\
		} else {						\
			tmp = PSCNV_RB_LEFT(parent, field);			\
			if (PSCNV_RB_COLOR(tmp, field) == PSCNV_RB_RED) {		\
				PSCNV_RB_SET_BLACKRED(tmp, parent, field);	\
				PSCNV_RB_ROTATE_RIGHT(head, parent, tmp, field);\
				tmp = PSCNV_RB_LEFT(parent, field);		\
			}						\
			if ((PSCNV_RB_LEFT(tmp, field) == NULL ||		\
			    PSCNV_RB_COLOR(PSCNV_RB_LEFT(tmp, field), field) == PSCNV_RB_BLACK) &&\
			    (PSCNV_RB_RIGHT(tmp, field) == NULL ||		\
			    PSCNV_RB_COLOR(PSCNV_RB_RIGHT(tmp, field), field) == PSCNV_RB_BLACK)) {\
				PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_RED;		\
				elm = parent;				\
				parent = PSCNV_RB_PARENT(elm, field);		\
			} else {					\
				if (PSCNV_RB_LEFT(tmp, field) == NULL ||	\
				    PSCNV_RB_COLOR(PSCNV_RB_LEFT(tmp, field), field) == PSCNV_RB_BLACK) {\
					struct type *oright;		\
					if ((oright = PSCNV_RB_RIGHT(tmp, field)) \
					    != NULL)			\
						PSCNV_RB_COLOR(oright, field) = PSCNV_RB_BLACK;\
					PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_RED;	\
					PSCNV_RB_ROTATE_LEFT(head, tmp, oright, field);\
					tmp = PSCNV_RB_LEFT(parent, field);	\
				}					\
				PSCNV_RB_COLOR(tmp, field) = PSCNV_RB_COLOR(parent, field);\
				PSCNV_RB_COLOR(parent, field) = PSCNV_RB_BLACK;	\
				if (PSCNV_RB_LEFT(tmp, field))		\
					PSCNV_RB_COLOR(PSCNV_RB_LEFT(tmp, field), field) = PSCNV_RB_BLACK;\
				PSCNV_RB_ROTATE_RIGHT(head, parent, tmp, field);\
				elm = PSCNV_RB_ROOT(head);			\
				break;					\
			}						\
		}							\
	}								\
	if (elm)							\
		PSCNV_RB_COLOR(elm, field) = PSCNV_RB_BLACK;			\
}									\
									\
attr struct type *							\
name##_PSCNV_RB_REMOVE(struct name *head, struct type *elm)			\
{									\
	struct type *child, *parent, *old = elm;			\
	int color;							\
	if (PSCNV_RB_LEFT(elm, field) == NULL)				\
		child = PSCNV_RB_RIGHT(elm, field);				\
	else if (PSCNV_RB_RIGHT(elm, field) == NULL)				\
		child = PSCNV_RB_LEFT(elm, field);				\
	else {								\
		struct type *left;					\
		elm = PSCNV_RB_RIGHT(elm, field);				\
		while ((left = PSCNV_RB_LEFT(elm, field)) != NULL)		\
			elm = left;					\
		child = PSCNV_RB_RIGHT(elm, field);				\
		parent = PSCNV_RB_PARENT(elm, field);				\
		color = PSCNV_RB_COLOR(elm, field);				\
		if (child)						\
			PSCNV_RB_PARENT(child, field) = parent;		\
		if (parent) {						\
			if (PSCNV_RB_LEFT(parent, field) == elm)		\
				PSCNV_RB_LEFT(parent, field) = child;		\
			else						\
				PSCNV_RB_RIGHT(parent, field) = child;	\
			PSCNV_RB_AUGMENT(parent);				\
		} else							\
			PSCNV_RB_ROOT(head) = child;				\
		if (PSCNV_RB_PARENT(elm, field) == old)			\
			parent = elm;					\
		(elm)->field = (old)->field;				\
		if (PSCNV_RB_PARENT(old, field)) {				\
			if (PSCNV_RB_LEFT(PSCNV_RB_PARENT(old, field), field) == old)\
				PSCNV_RB_LEFT(PSCNV_RB_PARENT(old, field), field) = elm;\
			else						\
				PSCNV_RB_RIGHT(PSCNV_RB_PARENT(old, field), field) = elm;\
			PSCNV_RB_AUGMENT(PSCNV_RB_PARENT(old, field));		\
		} else							\
			PSCNV_RB_ROOT(head) = elm;				\
		PSCNV_RB_PARENT(PSCNV_RB_LEFT(old, field), field) = elm;		\
		if (PSCNV_RB_RIGHT(old, field))				\
			PSCNV_RB_PARENT(PSCNV_RB_RIGHT(old, field), field) = elm;	\
		if (parent) {						\
			left = parent;					\
			do {						\
				PSCNV_RB_AUGMENT(left);			\
			} while ((left = PSCNV_RB_PARENT(left, field)) != NULL); \
		}							\
		goto color;						\
	}								\
	parent = PSCNV_RB_PARENT(elm, field);					\
	color = PSCNV_RB_COLOR(elm, field);					\
	if (child)							\
		PSCNV_RB_PARENT(child, field) = parent;			\
	if (parent) {							\
		if (PSCNV_RB_LEFT(parent, field) == elm)			\
			PSCNV_RB_LEFT(parent, field) = child;			\
		else							\
			PSCNV_RB_RIGHT(parent, field) = child;		\
		PSCNV_RB_AUGMENT(parent);					\
	} else								\
		PSCNV_RB_ROOT(head) = child;					\
color:									\
	if (color == PSCNV_RB_BLACK)						\
		name##_PSCNV_RB_REMOVE_COLOR(head, parent, child);		\
	return (old);							\
}									\
									\
/* Inserts a node into the RB tree */					\
attr struct type *							\
name##_PSCNV_RB_INSERT(struct name *head, struct type *elm)			\
{									\
	struct type *tmp;						\
	struct type *parent = NULL;					\
	int comp = 0;							\
	tmp = PSCNV_RB_ROOT(head);						\
	while (tmp) {							\
		parent = tmp;						\
		comp = (cmp)(elm, parent);				\
		if (comp < 0)						\
			tmp = PSCNV_RB_LEFT(tmp, field);			\
		else if (comp > 0)					\
			tmp = PSCNV_RB_RIGHT(tmp, field);			\
		else							\
			return (tmp);					\
	}								\
	PSCNV_RB_SET(elm, parent, field);					\
	if (parent != NULL) {						\
		if (comp < 0)						\
			PSCNV_RB_LEFT(parent, field) = elm;			\
		else							\
			PSCNV_RB_RIGHT(parent, field) = elm;			\
		PSCNV_RB_AUGMENT(parent);					\
	} else								\
		PSCNV_RB_ROOT(head) = elm;					\
	name##_PSCNV_RB_INSERT_COLOR(head, elm);				\
	return (NULL);							\
}									\
									\
/* Finds the node with the same key as elm */				\
attr struct type *							\
name##_PSCNV_RB_FIND(struct name *head, struct type *elm)			\
{									\
	struct type *tmp = PSCNV_RB_ROOT(head);				\
	int comp;							\
	while (tmp) {							\
		comp = cmp(elm, tmp);					\
		if (comp < 0)						\
			tmp = PSCNV_RB_LEFT(tmp, field);			\
		else if (comp > 0)					\
			tmp = PSCNV_RB_RIGHT(tmp, field);			\
		else							\
			return (tmp);					\
	}								\
	return (NULL);							\
}									\
									\
/* Finds the first node greater than or equal to the search key */	\
attr struct type *							\
name##_PSCNV_RB_NFIND(struct name *head, struct type *elm)			\
{									\
	struct type *tmp = PSCNV_RB_ROOT(head);				\
	struct type *res = NULL;					\
	int comp;							\
	while (tmp) {							\
		comp = cmp(elm, tmp);					\
		if (comp < 0) {						\
			res = tmp;					\
			tmp = PSCNV_RB_LEFT(tmp, field);			\
		}							\
		else if (comp > 0)					\
			tmp = PSCNV_RB_RIGHT(tmp, field);			\
		else							\
			return (tmp);					\
	}								\
	return (res);							\
}									\
									\
/* ARGSUSED */								\
attr struct type *							\
name##_PSCNV_RB_NEXT(struct type *elm)					\
{									\
	if (PSCNV_RB_RIGHT(elm, field)) {					\
		elm = PSCNV_RB_RIGHT(elm, field);				\
		while (PSCNV_RB_LEFT(elm, field))				\
			elm = PSCNV_RB_LEFT(elm, field);			\
	} else {							\
		if (PSCNV_RB_PARENT(elm, field) &&				\
		    (elm == PSCNV_RB_LEFT(PSCNV_RB_PARENT(elm, field), field)))	\
			elm = PSCNV_RB_PARENT(elm, field);			\
		else {							\
			while (PSCNV_RB_PARENT(elm, field) &&			\
			    (elm == PSCNV_RB_RIGHT(PSCNV_RB_PARENT(elm, field), field)))\
				elm = PSCNV_RB_PARENT(elm, field);		\
			elm = PSCNV_RB_PARENT(elm, field);			\
		}							\
	}								\
	return (elm);							\
}									\
									\
/* ARGSUSED */								\
attr struct type *							\
name##_PSCNV_RB_PREV(struct type *elm)					\
{									\
	if (PSCNV_RB_LEFT(elm, field)) {					\
		elm = PSCNV_RB_LEFT(elm, field);				\
		while (PSCNV_RB_RIGHT(elm, field))				\
			elm = PSCNV_RB_RIGHT(elm, field);			\
	} else {							\
		if (PSCNV_RB_PARENT(elm, field) &&				\
		    (elm == PSCNV_RB_RIGHT(PSCNV_RB_PARENT(elm, field), field)))	\
			elm = PSCNV_RB_PARENT(elm, field);			\
		else {							\
			while (PSCNV_RB_PARENT(elm, field) &&			\
			    (elm == PSCNV_RB_LEFT(PSCNV_RB_PARENT(elm, field), field)))\
				elm = PSCNV_RB_PARENT(elm, field);		\
			elm = PSCNV_RB_PARENT(elm, field);			\
		}							\
	}								\
	return (elm);							\
}									\
									\
attr struct type *							\
name##_PSCNV_RB_MINMAX(struct name *head, int val)				\
{									\
	struct type *tmp = PSCNV_RB_ROOT(head);				\
	struct type *parent = NULL;					\
	while (tmp) {							\
		parent = tmp;						\
		if (val < 0)						\
			tmp = PSCNV_RB_LEFT(tmp, field);			\
		else							\
			tmp = PSCNV_RB_RIGHT(tmp, field);			\
	}								\
	return (parent);						\
}

#define PSCNV_RB_NEGINF	-1
#define PSCNV_RB_INF	1

#define PSCNV_RB_INSERT(name, x, y)	name##_PSCNV_RB_INSERT(x, y)
#define PSCNV_RB_REMOVE(name, x, y)	name##_PSCNV_RB_REMOVE(x, y)
#define PSCNV_RB_FIND(name, x, y)	name##_PSCNV_RB_FIND(x, y)
#define PSCNV_RB_NFIND(name, x, y)	name##_PSCNV_RB_NFIND(x, y)
#define PSCNV_RB_NEXT(name, x, y)	name##_PSCNV_RB_NEXT(y)
#define PSCNV_RB_PREV(name, x, y)	name##_PSCNV_RB_PREV(y)
#define PSCNV_RB_MIN(name, x)		name##_PSCNV_RB_MINMAX(x, PSCNV_RB_NEGINF)
#define PSCNV_RB_MAX(name, x)		name##_PSCNV_RB_MINMAX(x, PSCNV_RB_INF)

#define PSCNV_RB_FOREACH(x, name, head)					\
	for ((x) = PSCNV_RB_MIN(name, head);					\
	     (x) != NULL;						\
	     (x) = name##_PSCNV_RB_NEXT(x))

#define PSCNV_RB_FOREACH_REVERSE(x, name, head)				\
	for ((x) = PSCNV_RB_MAX(name, head);					\
	     (x) != NULL;						\
	     (x) = name##_PSCNV_RB_PREV(x))

#endif	/* __PSCNV_TREE_H__ */
