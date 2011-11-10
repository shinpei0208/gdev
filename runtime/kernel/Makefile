# Makefile

CC		= gcc
TARGET		= libgdev
CFLAGS          = -O3 -Wall
HEADERS		= {gdev_api.h,gdev_lib.h,gdev_nvidia_def.h}

OBJS 	= $(patsubst %.c,%.o,$(wildcard ./*.c))
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)
	$(CC) -shared -Wl,-soname,$(TARGET).so.1 -o ./$(TARGET).so.1.0.0 $(OBJS) 
	ldconfig -n ./

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

install:
	cp -f ./$(TARGET).so.1 /usr/lib64/$(TARGET).so.1
	ln -sf /usr/lib64/$(TARGET).so.1 /usr/lib64/$(TARGET).so
	cp -f ./$(HEADERS) /usr/local/gdev/include

uninstall:
	rm -f /usr/lib64/$(TARGET).*
	rm -f /usr/local/gdev/include/$(HEADERS)

clean:
	rm -f $(TARGET).so.* $(OBJS) $(ZOMBIE)

