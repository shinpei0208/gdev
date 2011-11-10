# Makefile

CC	= gcc
CFLAGS	= -lgdev -I/usr/local/gdev/include

SRC  	= $(wildcard ./*.c)
OBJS 	= $(patsubst %.c,%.o,$(SRC))
ZOMBIE  = $(wildcard *~)

.PHONY: clean user_test

all: user_test

user_test: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o:%.c
	$(CC) -c $^ -o $@ $(CFLAGS)

clean:
	rm -f user_test $(OBJS) $(ZOMBIE)

