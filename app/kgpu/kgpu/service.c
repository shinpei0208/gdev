/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * GPU service management.
 *
 * link with -ldl
 */
#include <string.h>
#include <dlfcn.h>
#include <stdio.h>
#include <glob.h>
#include <stdlib.h>
#include "list.h"
#include "helper.h"

struct _kgpu_sitem {
    struct kgpu_service *s;
    void* libhandle;
    struct list_head list;
};

LIST_HEAD(services);

static struct _kgpu_sitem *lookup_kgpu_sitem(const char *name)
{
    struct _kgpu_sitem *i;
    struct list_head *e;
    
    if (!name)
	return NULL;

    list_for_each(e, &services) {
	i = list_entry(e, struct _kgpu_sitem, list);
	if (!strncmp(name, i->s->name, KGPU_SERVICE_NAME_SIZE))
	    return i;
    }

    return NULL;    
}

struct kgpu_service *kh_lookup_service(const char *name)
{
    struct _kgpu_sitem *i = lookup_kgpu_sitem(name);
    if (!i)
	return NULL;
    return i->s;
}

int kh_register_service(struct kgpu_service *s, void *libhandle)
{
    struct _kgpu_sitem *i;

    if (!s)
	return 1;
    i = (struct _kgpu_sitem *)malloc(sizeof(struct _kgpu_sitem));
    if (!i)
	return 1;

    i->s = s;
    i->libhandle = libhandle;
    INIT_LIST_HEAD(&i->list);

    list_add_tail(&i->list, &services);

    return 0;
}

static int __unregister_service(struct _kgpu_sitem *i)
{
    if (!i)
	return 1;

    list_del(&i->list);
    free(i);

    return 0;
}

int kh_unregister_service(const char *name)
{
    return __unregister_service(lookup_kgpu_sitem(name));    
}

int kh_load_service(const char *libpath)
{
    void *lh;
    fn_init_service init;
    char *err;
    int r=1;
    
    lh = dlopen(libpath, RTLD_LAZY);
    if (!lh)
    {
	fprintf(stderr,
		"Warning: open %s error, %s\n",
		libpath, dlerror());
    } else {
	init = (fn_init_service)dlsym(lh, SERVICE_INIT);
	if (!init)
	{
	    fprintf(stderr,
		    "Warning: %s has no service %s\n",
		    libpath, ((err=dlerror()) == NULL?"": err));
	    dlclose(lh);
	} else {
	    if (init(lh, kh_register_service))
	    {
		fprintf(stderr,
			"Warning: %s failed to register service\n",
			libpath);
		dlclose(lh);
	    } else
		r = 0;
	}	    
    }

    return r;
}

int kh_load_all_services(const char *dir)
{
    char path[256];
    int i;
    char *libpath;
    int e=0;

    glob_t glb = {0,NULL,0};

    snprintf(path, 256, "%s/%s*", dir, SERVICE_LIB_PREFIX);
    glob(path, 0, NULL, &glb);

    for (i=0; i<glb.gl_pathc; i++)
    {
	libpath = glb.gl_pathv[i];
	e += kh_load_service(libpath);
    }

    globfree(&glb);
    return e;
}

static int __unload_service(struct _kgpu_sitem* i)
{
    void *lh;
    fn_finit_service finit;
    int r=1;
    if (!i)
	return 1;

    lh = i->libhandle;
    
    if (lh) {
	finit = (fn_finit_service)dlsym(lh, SERVICE_FINIT);
	if (finit)
	{
	    if (finit(lh, kh_unregister_service))
	    {
		fprintf(stderr,
			"Warning: failed to unregister service %s\n",
			i->s->name);
	    } else
		r = 0;
	} else {
	    __unregister_service(i);
	    r = 0;
	}
	
	dlclose(lh);
    } else {
	__unregister_service(i);
	r=0;
    }

    return r;
}

int kh_unload_service(const char *name)
{
    return __unload_service(lookup_kgpu_sitem(name));
}

int kh_unload_all_services()
{
    struct list_head *p, *n;
    int e=0;

    list_for_each_safe(p, n, &services) {
	e += __unload_service(list_entry(p, struct _kgpu_sitem, list));
    }
    return e;
}
