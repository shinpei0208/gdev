# Makefile

CC		= gcc
CXX		= g++
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS        = -std=c++0x

OBJS 	= AddLocationMetadata.o AffineLinearScan.o AffineRegister.o
OBJS   += AggressivePredication.o AssignFallThroughEdge.o
OBJS   += CoalescedRegister.o ConstantPropagationPass.o
OBJS   += ConvertPredicationToSelectPass.o DeadCodeEliminationPass.o
OBJS   += DefaultLayoutPass.o DivergenceLinearScan.o DivergenceRegister.o
OBJS   += EnforceLockStepExecutionPass.o FunctionInliningPass.o
OBJS   += GlobalValueNumberingPass.o HoistParameterLoadsPass.o
OBJS   += HoistSpecialValueDefinitionsPass.o IPDOMReconvergencePass.o
OBJS   += KernelDrawerPass.o LinearScanRegisterAllocationPass.o
OBJS   += LoopUnrollingPass.o MIMDThreadSchedulingPass.o MemoryArray.o
OBJS   += ModuleLinkerPass.o MoveEliminationPass.o Pass.o PassFactory.o
OBJS   += PassManager.o PriorityLayoutPass.o ReadableLayoutPass.o
OBJS   += RemoveBarrierPass.o SharedPtrAttribute.o
OBJS   += SimplifyControlFlowGraphPass.o
OBJS   += SpillPolicy.o SplitBasicBlockPass.o StructuralTransform.o
OBJS   += SubkernelFormationPass.o SyncEliminationPass.o
OBJS   += ThreadFrontierReconvergencePass.o
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

