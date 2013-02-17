# Makefile

include Driver.mk

CC = gcc
TARGET = libgdev
CFLAGS = -O2 -Wall $(EXTRA_CFLAGS)
GDEVDIR	= /usr/local/gdev
HEADERS	= gdev_api.h gdev_nvidia_def.h gdev_list.h gdev_time.h

nvrm_OBJS = nvrm_gdev.o nvrm.o ioctl.o mthd.o handle.o channel.o memory.o

gdev_OBJS = gdev_gdev.o libgdev.o libgdev_ib.o
pscnv_OBJS = pscnv_gdev.o libpscnv.o libpscnv_ib.o
nouveau_OBJS = nouveau_gdev.o libnouveau.o libnouveau_ib.o

OBJS =	gdev_lib.o \
	gdev_api.o gdev_device.o gdev_sched.o \
	gdev_nvidia.o gdev_nvidia_fifo.o gdev_nvidia_compute.o gdev_nvidia_mem.o gdev_nvidia_shm.o gdev_nvidia_nvc0.o $($(DRIVER_NAME)_OBJS)

ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) $(LDFLAGS) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) $(EXTRA_LIBS)

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS) $(EXTRA_LIBS)

install:
	@if test ! -d $(DESTDIR)/$(GDEVDIR) ; then mkdir $(DESTDIR)/$(GDEVDIR) ; fi ;
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/lib64 ; then mkdir $(DESTDIR)/$(GDEVDIR)/lib64 ; fi ;
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/include ; then mkdir $(DESTDIR)/$(GDEVDIR)/include ; fi ;
	cp -f ./$(TARGET).so.1.0.0 $(DESTDIR)/$(GDEVDIR)/lib64/
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1.0.0 $(DESTDIR)/$(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(DESTDIR)/$(GDEVDIR)/lib64/$(TARGET).so
	cp -f $(HEADERS) $(DESTDIR)/$(GDEVDIR)/include

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/gdev*.h

clean:
	rm -f $(TARGET).so.* $(OBJS) $(ZOMBIE)

