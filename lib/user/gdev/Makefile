# Makefile

include Driver.mk

CC = gcc
TARGET = libgdev
CFLAGS = -O2 -Wall $(EXTRA_CFLAGS)
GDEVDIR	= /usr/local/gdev
HEADERS	= gdev_api.h gdev_nvidia_def.h gdev_list.h gdev_time.h

OBJS = 	$(DRIVER_NAME)_gdev.o lib$(DRIVER_NAME).o lib$(DRIVER_NAME)_ib.o \
	gdev_lib.o \
	gdev_api.o gdev_device.o gdev_sched.o \
	gdev_nvidia.o gdev_nvidia_compute.o gdev_nvidia_mem.o gdev_nvidia_shm.o gdev_nvidia_nvc0.o 

ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) $(EXTRA_LIBS)

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS) $(EXTRA_LIBS)

install:
	@if test ! -d $(GDEVDIR) ; then mkdir $(GDEVDIR) ; fi ;
	@if test ! -d $(GDEVDIR)/lib64 ; then mkdir $(GDEVDIR)/lib64 ; fi ;
	@if test ! -d $(GDEVDIR)/include ; then mkdir $(GDEVDIR)/include ; fi ;
	cp -f ./$(TARGET).so.1.0.0 $(GDEVDIR)/lib64/
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1.0.0 $(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(GDEVDIR)/lib64/$(TARGET).so
	cp -f $(HEADERS) $(GDEVDIR)/include

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/gdev*.h

clean:
	rm -f $(TARGET).so.* $(OBJS) $(ZOMBIE)

