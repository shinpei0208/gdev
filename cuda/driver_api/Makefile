# Makefile

CC		= gcc
TARGET		= libcuda
CUDUMP		= cudump
GDEVDIR		= /usr/local/gdev
CFLAGS          = -O3 -Wall -I$(GDEVDIR)/include

#OBJS 	= $(patsubst %.c,%.o,$(wildcard ./*.c))
OBJS 	= init.o device.o version.o context.o module.o execution.o memory.o \
	  gdev_cuda.o
CUDUMP_OBJS = cudump.o gdev_cuda.o
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) 
	$(CC) $(CFLAGS) -o ./$(CUDUMP) $(CUDUMP).c gdev_cuda.c

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

install:
	@if test ! -d $(GDEVDIR)/bin ; then mkdir $(GDEVDIR)/bin ; fi ;
	cp -f ./$(CUDUMP) $(GDEVDIR)/bin/
	cp -f ./$(TARGET).so.1.0.0 $(GDEVDIR)/lib64/
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1.0.0 $(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(GDEVDIR)/lib64/$(TARGET).so
	cp -f ./cuda.h $(GDEVDIR)/include

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/{$(HEADERS)}
	rm -f $(GDEVDIR)/bin/$(CUDUMP)

clean:
	rm -f $(TARGET).so.* $(CUDUMP) $(OBJS) $(CUDUMP_OBJS) $(ZOMBIE)

