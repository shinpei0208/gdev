# Makefile

#include ../../../API.mk

CC		= gcc
CXX		= g++
CFLAGS		= -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS	= -std=c++0x

ifeq ($(ENABLE_CUBIN_MODULE),1)
	CXXFLAGS += -DENABLE_CUBIN_MODULE
endif

OBJS		= CudaDriver.o CudaDriverFrontend.o
OBJS	       += CudaDriverInterface.o CudaRuntime.o CudaRuntimeInterface.o
OBJS	       += CudaWorkerThread.o FatBinaryContext.o cuda_runtime.o

ZOMBIE		= $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

