# Makefile

#include ../../../API.mk

CC		= gcc
CXX		= g++
CFLAGS		= -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS	= -std=c++0x

ifeq ($(ENABLE_CUBIN_MODULE),1)
	CXXFLAGS += -DENABLE_CUBIN_MODULE
endif

OBJS		= AddLocationMetadata.o AffineLinearScan.o AffineRegister.o
OBJS	       += AggressivePredication.o AssignFallThroughEdge.o
OBJS	       += CoalescedRegister.o ConstantPropagationPass.o
OBJS	       += ConvertPredicationToSelectPass.o DeadCodeEliminationPass.o
OBJS	       += DefaultLayoutPass.o DivergenceLinearScan.o
OBJS	       += DivergenceRegister.o EnforceLockStepExecutionPass.o
OBJS	       += FunctionInliningPass.o GlobalValueNumberingPass.o
OBJS	       += HoistParameterLoadsPass.o HoistSpecialValueDefinitionsPass.o
OBJS	       += IPDOMReconvergencePass.o KernelDrawerPass.o
OBJS	       += LinearScanRegisterAllocationPass.o LoopUnrollingPass.o
OBJS	       += MIMDThreadSchedulingPass.o MemoryArray.o
OBJS	       += ModuleLinkerPass.o MoveEliminationPass.o Pass.o PassFactory.o
OBJS	       += PassManager.o PriorityLayoutPass.o ReadableLayoutPass.o
OBJS	       += RemoveBarrierPass.o SharedPtrAttribute.o
OBJS	       += SimplifyControlFlowGraphPass.o SpillPolicy.o
OBJS	       += SplitBasicBlockPass.o StructuralTransform.o
OBJS	       += SubkernelFormationPass.o SyncEliminationPass.o
OBJS	       += ThreadFrontierReconvergencePass.o

ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

