# Makefile

#include ../../../API.mk

GDEVDIR	       ?= /usr/local/gdev

CC		= gcc
CXX		= g++
CFLAGS		= -O3 -Wall -I $(GDEVDIR)/include -I ../.. -I ../../..
CXXFLAGS	= -std=c++0x

ifeq ($(ENABLE_CUBIN_MODULE),1)
	CXXFLAGS += -DENABLE_CUBIN_MODULE
endif

OBJS		= ControlFlowGraph.o Dim3.o Global.o IRKernel.o Instruction.o
OBJS	       += Kernel.o Local.o Module.o PTXInstruction.o PTXKernel.o
OBJS	       += PTXOperand.o PTXStatement.o Parameter.o Texture.o

ZOMBIE		= $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

implementation/Module.cpp: ../../ptxgrammar.hpp

../../ptxgrammar.hpp:
	$(MAKE) -C ../parser/ $@

clean:
	rm -f $(OBJS) $(ZOMBIE)

