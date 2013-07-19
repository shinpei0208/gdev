# Makefile

CC	= gcc
TARGET	= libgdev
CFLAGS	= -O3 -Wall
GDEVDIR	= /usr/local/gdev
HEADERS	= gdev_api.h gdev_nvidia_def.h gdev_list.h gdev_time.h

#OBJS 	= $(patsubst %.c,%.o,$(wildcard ./*.c))
OBJS 	= gdev_lib.o
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) 

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

install:
	@if test ! -d $(GDEVDIR) ; then mkdir $(GDEVDIR) ; fi ;
	@if test ! -d $(GDEVDIR)/lib64 ; then mkdir $(GDEVDIR)/lib64 ; fi ;
	@if test ! -d $(GDEVDIR)/include ; then mkdir $(GDEVDIR)/include ; fi ;
	cp -f ./$(TARGET).so.1.0.0 $(GDEVDIR)/lib64/
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1.0.0 $(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(GDEVDIR)/lib64/$(TARGET).so
	cp -f $(HEADERS) $(GDEVDIR)/include
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
