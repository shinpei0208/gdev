# Makefile

CC		= gcc
CXX		= g++
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS        = -std=c++0x

OBJS	= ControlFlowGraph.o Dim3.o Global.o IRKernel.o Instruction.o
OBJS   += Kernel.o Local.o Module.o PTXInstruction.o PTXKernel.o
OBJS   += PTXOperand.o PTXStatement.o Parameter.o Texture.o

ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

