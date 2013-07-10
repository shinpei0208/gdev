# Makefile

CC		= gcc
CXX		= g++
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS        = -std=c++0x

OBJS 	= ocelot.o
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

