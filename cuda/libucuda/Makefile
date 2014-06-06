include API.mk

CC		= gcc
TARGET		= libucuda
CUDUMP		= cudump
GDEVDIR		= /usr/local/gdev
CFLAGS		= -O3 -Wall -I $(GDEVDIR)/include
LDFLAGS		= -L $(GDEVDIR)/lib64 -lgdev -lrt -Wl,-Bsymbolic
LDFLAGS	       += -Wl,--no-as-needed

#OBJS 		= $(patsubst %.c,%.o,$(wildcard ./*.c))
OBJS 		= init.o device.o version.o context.o module.o execution.o
OBJS	       += memory.o stream.o event.o gdev_cuda.o dummy.o
OBJS	       += extension/memmap.o extension/ipc.o

CUDUMP_OBJS	= cudump.o gdev_cuda.o

ANALYSIS	= AffineAnalysis Analysis BranchInfo ControlTree
ANALYSIS       += ConvergentRegionAnalysis DataflowGraph DirectionalGraph
ANALYSIS       += DivergenceAnalysis DivergenceGraph DominatorTree
ANALYSIS       += LoopAnalysis PostdominatorTree PredicateAnalysis
ANALYSIS       += ProgramStructureGraph SSAGraph SimpleAliasAnalysis
ANALYSIS       += SparseAnalysis StructuralAnalysis SuperblockAnalysis
ANALYSIS       += ThreadFrontierAnalysis
ANALYSIS_OBJS	= $(patsubst %, analysis/%.o, $(ANALYSIS))

API		= ocelot
API_OBJS	= $(patsubst %, api/%.o, $(API))

CUDA		= CudaDriver CudaDriverFrontend
CUDA	       += CudaDriverInterface CudaRuntime CudaRuntimeInterface
CUDA	       += CudaWorkerThread FatBinaryContext cuda_runtime
CUDA_OBJS	= $(patsubst %, cuda/%.o, $(CUDA))

EXECUTIVE	= Device FrameInfo ExecutableKernel
EXECUTIVE      += NVIDIAExecutableKernel NVIDIAGPUDevice
EXECUTIVE      += RuntimeException 
EXECUTIVE_OBJS	= $(patsubst %, executive/%.o, $(EXECUTIVE))

IR		= ControlFlowGraph Dim3 Global IRKernel Instruction
IR	       += Kernel Local Module PTXInstruction PTXKernel
IR	       += PTXOperand PTXStatement Parameter Texture
IR_OBJS		= $(patsubst %, ir/%.o, $(IR))

PARSER		= PTXLexer PTXParser ptx ptxgrammar
PARSER_OBJS	= $(patsubst %, parser/%.o, $(PARSER))

TRANSFORMS	= AddLocationMetadata AffineLinearScan AffineRegister
TRANSFORMS     += AggressivePredication AssignFallThroughEdge
TRANSFORMS     += CoalescedRegister ConstantPropagationPass
TRANSFORMS     += ConvertPredicationToSelectPass DeadCodeEliminationPass
TRANSFORMS     += DefaultLayoutPass DivergenceLinearScan
TRANSFORMS     += DivergenceRegister EnforceLockStepExecutionPass
TRANSFORMS     += FunctionInliningPass GlobalValueNumberingPass
TRANSFORMS     += HoistParameterLoadsPass HoistSpecialValueDefinitionsPass
TRANSFORMS     += IPDOMReconvergencePass KernelDrawerPass
TRANSFORMS     += LinearScanRegisterAllocationPass LoopUnrollingPass
TRANSFORMS     += MIMDThreadSchedulingPass MemoryArray
TRANSFORMS     += ModuleLinkerPass MoveEliminationPass Pass PassFactory
TRANSFORMS     += PassManager PriorityLayoutPass ReadableLayoutPass
TRANSFORMS     += RemoveBarrierPass SharedPtrAttribute
TRANSFORMS     += SimplifyControlFlowGraphPass
TRANSFORMS     += SpillPolicy SplitBasicBlockPass StructuralTransform
TRANSFORMS     += SubkernelFormationPass SyncEliminationPass
TRANSFORMS     += ThreadFrontierReconvergencePass
TRANSFORMS_OBJS	= $(patsubst %, transforms/%.o, $(TRANSFORMS))

OCELOT_OBJS	= $(patsubst %, runtime/ocelot/%, $(ANALYSIS_OBJS))
OCELOT_OBJS    += $(patsubst %, runtime/ocelot/%, $(API_OBJS))
OCELOT_OBJS    += $(patsubst %, runtime/ocelot/%, $(CUDA_OBJS))
OCELOT_OBJS    += $(patsubst %, runtime/ocelot/%, $(EXECUTIVE_OBJS))
OCELOT_OBJS    += $(patsubst %, runtime/ocelot/%, $(IR_OBJS))
OCELOT_OBJS    += $(patsubst %, runtime/ocelot/%, $(PARSER_OBJS))
OCELOT_OBJS    += $(patsubst %, runtime/ocelot/%, $(TRANSFORMS_OBJS))

HYDRAZINE	= ELFFile Exception LowLevelTimer SystemCompatibility
HYDRAZINE      += Thread Timer Version compression debug string
HYDRAZINE_OBJS	= $(patsubst %, runtime/hydrazine/%.o, $(HYDRAZINE))

ZOMBIE		= $(wildcard ./*~)

ifeq ($(CUDA_RUNTIME_API),1)
	OBJS += $(OCELOT_OBJS) $(HYDRAZINE_OBJS)
	RUNTIME_LDFLAGS = -lboost_thread -ldl
endif

all: $(TARGET) $(CUDUMP)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(RUNTIME_LDFLAGS) -shared -Wl,-soname,$(TARGET).so.1 \
		-o ./$(TARGET).so.1.0.0 $(OBJS) 

$(CUDUMP): $(CUDUMP_OBJS)
	$(CC) $(LDFLAGS) -o ./$(CUDUMP) $(CUDUMP_OBJS)

%.o:%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

%.runtime:
	$(MAKE) -C runtime $(basename $@)

$(OCELOT_OBJS):
	$(MAKE) ocelot.runtime

$(HYDRAZINE_OBJS):
	$(MAKE) hydrazine.runtime

install:
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/bin ; then mkdir -p $(DESTDIR)/$(GDEVDIR)/bin ; fi ;
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/lib64 ; then mkdir -p $(DESTDIR)/$(GDEVDIR)/lib64 ; fi ;
	@if test ! -d $(DESTDIR)/$(GDEVDIR)/include ; then mkdir -p $(DESTDIR)/$(GDEVDIR)/include ; fi ; cp -f ./$(CUDUMP) $(DESTDIR)/$(GDEVDIR)/bin/
	cp -f ./$(TARGET).so.1.0.0 $(DESTDIR)/$(GDEVDIR)/lib64/
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1.0.0 $(DESTDIR)/$(GDEVDIR)/lib64/$(TARGET).so.1
	ln -sf $(GDEVDIR)/lib64/$(TARGET).so.1 $(DESTDIR)/$(GDEVDIR)/lib64/$(TARGET).so
	cp -f ./cuda.h $(DESTDIR)/$(GDEVDIR)/include

uninstall:
	rm -f $(GDEVDIR)/lib64/$(TARGET).*
	rm -f $(GDEVDIR)/include/{$(HEADERS)}
	rm -f $(GDEVDIR)/bin/$(CUDUMP)

clean: clean.runtime
	rm -f $(TARGET).so.* $(CUDUMP) $(OBJS) $(CUDUMP_OBJS) $(ZOMBIE)

