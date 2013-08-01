# Makefile

#include ../../../API.mk

CC		= gcc
CXX		= g++
CFLAGS		= -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS	= -std=c++0x

ifeq ($(ENABLE_CUBIN_MODULE),1)
	CXXFLAGS += -DENABLE_CUBIN_MODULE
endif

OBJS		= AffineAnalysis.o Analysis.o BranchInfo.o ControlTree.o
OBJS	       += ConvergentRegionAnalysis.o DataflowGraph.o DirectionalGraph.o
OBJS	       += DivergenceAnalysis.o DivergenceGraph.o DominatorTree.o
OBJS	       += LoopAnalysis.o PostdominatorTree.o PredicateAnalysis.o
OBJS	       += ProgramStructureGraph.o SSAGraph.o SimpleAliasAnalysis.o
OBJS	       += SparseAnalysis.o StructuralAnalysis.o SuperblockAnalysis.o
OBJS	       += ThreadFrontierAnalysis.o

ZOMBIE		= $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

