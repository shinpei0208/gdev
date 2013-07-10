# Makefile

CC		= gcc
CXX		= g++
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS        = -std=c++0x

OBJS	= Device.o FrameInfo.o ExecutableKernel.o
OBJS   += NVIDIAExecutableKernel.o NVIDIAGPUDevice.o
OBJS   += RuntimeException.o


ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

