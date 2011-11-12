# Makefile

CC		= gcc
TARGET		= libcuda
GDEVDIR		= /usr/local/gdev
CFLAGS          = -O3 -Wall -I$(GDEVDIR)/include
HEADERS		= {cuda.h}

OBJS 	= $(patsubst %.c,%.o,$(wildcard ./*.c))
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) 

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

install:
	cp -f ./$(TARGET).so.1 $(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(GDEVDIR)/lib64/$(TARGET).so
	cp -f ./$(HEADERS) $(GDEVDIR)/include

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/$(HEADERS)

clean:
	rm -f $(TARGET).so.* $(OBJS) $(ZOMBIE)

