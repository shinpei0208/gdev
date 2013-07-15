# Makefile

WITHOUT_RUNTIME ?= 1 #enable: 0, disable:1

CC		= gcc
TARGET		= libucuda
CUDUMP		= cudump
GDEVDIR		= /usr/local/gdev
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include
LDFLAGS = -L $(GDEVDIR)/lib64 -lgdev -lrt -Wl,-Bsymbolic

#OBJS 	= $(patsubst %.c,%.o,$(wildcard ./*.c))
OBJS 	= init.o device.o version.o context.o module.o execution.o memory.o stream.o event.o gdev_cuda.o dummy.o extension/memmap.o extension/ipc.o
CUDUMP_OBJS = cudump.o gdev_cuda.o
OCELOT_OBJS = runtime/ocelot/*/*.o
HYDRAZ_OBJS = runtime/hydrazine/*.o
ZOMBIE  = $(wildcard ./*~)

ifeq ($(WITHOUT_RUNTIME),0)
	OBJS += $(OCELOT_OBJS) $(HYDRAZ_OBJS)
	LDFLAGS += -lboost_thread -ldl
endif

all: $(OBJS)
	$(CC) $(LDFLAGS)  -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) 
	$(CC) $(LDFLAGS) $(CFLAGS) -o ./$(CUDUMP) $(CUDUMP).c gdev_cuda.c

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

%.runtime:
	$(MAKE) -C runtime $(basename $@)

$(OCELOT_OBJS):
	$(MAKE) ocelot.runtime

$(HYDRAZ_OBJS):
	$(MAKE) hydrazine.runtime

install:
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/bin ; then mkdir -p $(DESTDIR)/$(GDEVDIR)/bin ; fi ;
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/lib64 ; then mkdir -p $(DESTDIR)/$(GDEVDIR)/lib64 ; fi ;
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/include ; then mkdir -p $(DESTDIR)/$(GDEVDIR)/include ; fi ;
	cp -f ./$(CUDUMP) $(DESTDIR)/$(GDEVDIR)/bin/
	cp -f ./$(TARGET).so.1.0.0 $(DESTDIR)/$(GDEVDIR)/lib64/
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1.0.0 $(DESTDIR)/$(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(DESTDIR)/$(GDEVDIR)/lib64/$(TARGET).so
	cp -f ./cuda.h $(DESTDIR)/$(GDEVDIR)/include

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/{$(HEADERS)}
	rm -f $(GDEVDIR)/bin/$(CUDUMP)

clean: clean.runtime
	rm -f $(TARGET).so.* $(CUDUMP) $(OBJS) $(CUDUMP_OBJS) $(ZOMBIE)

