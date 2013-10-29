# Makefile

include Driver.mk
include Extra.mk

CC = gcc
TARGET = libgdev
CFLAGS = -O2 -Wall $(EXTRA_CFLAGS)
LDFLAGS = -Bsymbolic
GDEVDIR	= /usr/local/gdev
HEADERS	= gdev_api.h gdev_autogen.h gdev_nvidia_def.h gdev_list.h gdev_time.h

OBJS =	gdev_lib.o \
	gdev_api.o gdev_device.o gdev_sched.o \
	gdev_nvidia.o gdev_nvidia_fifo.o gdev_nvidia_compute.o gdev_nvidia_mem.o gdev_nvidia_shm.o gdev_nvidia_nvc0.o gdev_nvidia_nve4.o $(EXTRA_OBJS)
OBJSMON = gdev_usched_monitor.o gdev_usched_monitor_init.o


ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) $(LDFLAGS) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) $(EXTRA_LIBS)

sched: $(OBJSMON)
	$(CC) $(LDFLAGS) -lpthread $(OBJSMON) -o gdev_usched_monitor $(CFLAGS)

sched_install :
	cp -f gdev_usched_monitor $(DESTDIR)/$(GDEVDIR)/bin

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
	@make echo

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/gdev*.h

clean:
	rm -f $(TARGET).so.* $(OBJS) $(ZOMBIE)

echo:
	@echo ---------------------------------------------------------------
	@echo Gdev Library installation was completed! 
	@echo
	@echo *Please make sure your PATH includes /usr/local/gdev/bin
	@echo *Please make sure your LD_LIBRARY_PATH includes /usr/local/gdev/lib64
	@echo 
	@echo Please type follows commands':'
	@echo     export LD_LIBRARY_PATH='"'/usr/local/gdev/lib64:'$$'LD_LIBRARY_PATH'"'
	@echo     export PATH='"'/usr/local/gdev/bin:'$$'PATH'"'
	@echo 
	@echo If you are administrator, write to /etc/ld.so.conf and run ldconfig 
	@echo ---------------------------------------------------------------
