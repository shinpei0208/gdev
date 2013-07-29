# Makefile

#include ../../API.mk

CC		= gcc
CXX		= g++
TARGET		= hydrazine
CFLAGS 		= -O3 -Wall -I $(GDEVDIR)/include -I ..
CXXFLAGS	= -std=c++0x

OBJS		= ELFFile.o Exception.o LowLevelTimer.o SystemCompatibility.o
OBJS	       += Thread.o Timer.o Version.o compression.o debug.o string.o

ZOMBIE		= $(wildcard ./*~)

all: $(TARGET)

$(TARGET): $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

