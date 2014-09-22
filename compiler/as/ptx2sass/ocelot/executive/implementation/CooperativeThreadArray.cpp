/*! \file CooperativeThreadArray.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date 5 February 2009
	\brief defines the state of each cooperative thread array 
		with associated code for emulating its execution
*/

// Ocelot Includes
#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/Module.h>

#include <ocelot/executive/interface/RuntimeException.h>
#include <ocelot/executive/interface/CooperativeThreadArray.h>
#include <ocelot/executive/interface/EmulatedKernel.h>
#include <ocelot/executive/interface/EmulatedKernelScheduler.h>
#include <ocelot/executive/interface/CTAContext.h>
#include <ocelot/executive/interface/TextureOperations.h>

#include <ocelot/api/interface/OcelotConfiguration.h>

// Hydrazine Includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/FloatingPoint.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/math.h>

// Standard Library Includes
#include <cassert>
#include <cmath>
#include <cstring>
#include <climits>
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

// global control for enabling reporting within the emulator
#define REPORT_BASE 0

// reporting for kernel instructions
#define REPORT_STATIC_INSTRUCTIONS 0
#define REPORT_DYNAMIC_INSTRUCTIONS 1

// reporting for register accesses
#define REPORT_NTH_THREAD_ONLY 1
#define NTH_THREAD 0
#define REPORT_REGISTER_READS 1
#define REPORT_REGISTER_WRITES 1
#define REPORT_PREDICATE_READS 0

// individually turn on or off reporting for particular instructions
#define REPORT_ABS        1
#define REPORT_ADD        1
#define REPORT_ADDC       1
#define REPORT_AND        1
#define REPORT_ATOM       1
#define REPORT_BAR        1
#define REPORT_BRA        1
#define REPORT_BRKPT      1
#define REPORT_CALL       1
#define REPORT_CNOT       1
#define REPORT_COS        1
#define REPORT_CVT        1
#define REPORT_DIV        1
#define REPORT_EX2        1
#define REPORT_EXIT       1
#define REPORT_LD         0
#define REPORT_LG2        1
#define REPORT_MAD24      1
#define REPORT_MAD        1
#define REPORT_MAX        1
#define REPORT_MIN        1
#define REPORT_MOV        1
#define REPORT_MUL24      1
#define REPORT_MUL        1
#define REPORT_NEG        1
#define REPORT_NOT        1
#define REPORT_OR         1
#define REPORT_RCP        1
#define REPORT_RECONVERGE 0
#define REPORT_RED        1
#define REPORT_REM        1
#define REPORT_RET        1
#define REPORT_RSQRT      1
#define REPORT_SAD        1
#define REPORT_SELP       0
#define REPORT_SETP       0
#define REPORT_SET        1
#define REPORT_SHL        1
#define REPORT_SHR        1
#define REPORT_SIN        1
#define REPORT_SLCT       1
#define REPORT_SQRT       1
#define REPORT_ST         0
#define REPORT_SUB        1
#define REPORT_SUBC       1
#define REPORT_TEX        1
#define REPORT_TRAP       1	
#define REPORT_VOTE       1
#define REPORT_XOR        1

template <typename T>
static T CTAAbs(T a) {
	if (a < 0) {
		return -a;
	}
	return a;
}

template<typename T>
bool issubnormal(T r0)
{
	return !hydrazine::isnormal(r0) && !hydrazine::isnan(r0)
		&& !hydrazine::isinf(r0) && r0 != (T)0;
}

static executive::ReconvergenceMechanism*
	getReconvergenceMechanism(executive::CooperativeThreadArray* cta) {
	
	typedef api::OcelotConfiguration config;

	if (config::get().executive.reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_IPDOM) {
		return new executive::ReconvergenceIPDOM(cta);
	}
	else if (config::get().executive.reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_Barrier) {
		return new executive::ReconvergenceBarrier(cta);
	}
	else if (config::get().executive.reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_TFGen6) {
		return new executive::ReconvergenceTFGen6(cta);
	}
	else if (config::get().executive.reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_TFSortedStack) {
		return new executive::ReconvergenceTFSortedStack(cta);
	}
	else if (config::get().executive.reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_TFSoftware) {
		return new executive::ReconvergenceTFSoftware(cta);
	}
	else {
		assertM(false, "unknown thread reconvergence mechanism - "
			<< config::get().executive.reconvergenceMechanism);
	}

	return 0;
}

#define max(a, b) ((a) < (b) ? (b) : (a))
#define min(a, b) ((a) > (b) ? (b) : (a))

//////////////////////////////////////////////////////////////////////////////////////////////////
/*!
	Constructs a cooperative thread array from an EmulatedKernel instance

	\param kernel pointer to EmulatedKernel to which this CTA belongs
*/
executive::CooperativeThreadArray::CooperativeThreadArray(
	EmulatedKernel *k, const ir::Dim3& grid, bool trace):
	blockDim(k->blockDim()),
	gridDim(grid),
	threadCount(blockDim.x*blockDim.y*blockDim.z),
	kernel(k),
	functionCallStack(blockDim.x*blockDim.y*blockDim.z, k->argumentMemorySize(), 
		k->parameterMemorySize(), k->registerCount(), k->localMemorySize(),
		k->globalLocalMemorySize(), k->totalSharedMemorySize()),
	clock(0),
	traceEvents(trace) {

	reconvergenceMechanism = getReconvergenceMechanism(this);

	initialize(blockDim);
}

executive::CooperativeThreadArray::CooperativeThreadArray(
	EmulatedKernel *k, const ir::Dim3& grid, const ir::Dim3& cta,
	unsigned int argumentMemorySize, unsigned int parameterMemorySize,
	unsigned int registerCount, unsigned int localMemorySize,
	unsigned int globalLocalMemorySize, unsigned int totalSharedMemorySize,
	bool trace):
	blockDim(cta),
	gridDim(grid),
	threadCount(blockDim.x*blockDim.y*blockDim.z),
	kernel(k),
	functionCallStack(blockDim.x*blockDim.y*blockDim.z, argumentMemorySize, 
		parameterMemorySize, registerCount, localMemorySize,
		globalLocalMemorySize, totalSharedMemorySize),
	clock(0),
	traceEvents(trace) {

	reconvergenceMechanism = getReconvergenceMechanism(this);

	initialize(blockDim);
}

executive::CooperativeThreadArray::CooperativeThreadArray() : kernel(0),
	reconvergenceMechanism(0) {
}

/*!
	Destroys state associated with CTA
*/
executive::CooperativeThreadArray::~CooperativeThreadArray() {
	delete reconvergenceMechanism;
}

/*!
	Gets current instruction
*/
const ir::PTXInstruction&
	executive::CooperativeThreadArray::currentInstruction(
	CTAContext & context) {
	return kernel->instructions[context.PC];
}

ir::PTXU32 executive::CooperativeThreadArray::getSpecialValue( 
	const int threadId, const ir::PTXOperand::SpecialRegister reg, 
	const ir::PTXOperand::VectorIndex index ) const
{
	assert( reg != ir::PTXOperand::SpecialRegister_invalid );
	assert( reg != ir::PTXOperand::pm0 );
	assert( reg != ir::PTXOperand::pm1 );
	assert( reg != ir::PTXOperand::pm2 );
	assert( reg != ir::PTXOperand::pm3 );
	assert( reg != ir::PTXOperand::smId );
	assert( reg != ir::PTXOperand::nsmId );
	assert( reg != ir::PTXOperand::gridId );

	//
	// some special registers are not warp-agnostic, so we give the option to specify
	// a constant-size warp in the Ocelot configuration. If this value is 0 or larger
	// than the CTA size, the warpSize is assumed to be the CTA size
	//
	int warpSize = api::OcelotConfiguration::get().executive.warpSize;
	if (warpSize == 0 || warpSize > threadCount) {
		warpSize = threadCount;
	}

	switch( reg ) {
		case ir::PTXOperand::laneId: {
			return (threadId % warpSize);
		}
		case ir::PTXOperand::lanemask_eq: {	
			return (1 << (threadId % warpSize));
		}
		case ir::PTXOperand::lanemask_lt: // fall through
		case ir::PTXOperand::lanemask_le: {
			// we must assume this->currentEvent.active has been updated
			unsigned int mask = (reg == ir::PTXOperand::lanemask_le ?
				(1 << (threadId % warpSize)) : 0);
			int baseId = 0;
			for (int tWarpId = (threadId / warpSize);
				tWarpId < threadId; ++tWarpId) {
				mask |= ((currentEvent.active[tWarpId] ? 1 : 0) << baseId);
				++baseId;
			}
			return mask;
		}
		case ir::PTXOperand::lanemask_gt: // fall through
		case ir::PTXOperand::lanemask_ge: {
			// we must assume this->currentEvent.active has been updated
			unsigned int mask = (reg == ir::PTXOperand::lanemask_ge ?
				(1 << (threadId % warpSize)) : 0);
			int endThreadId = (threadId + warpSize - 1)/warpSize;
			for (int tWarpId = threadId + 1; tWarpId < endThreadId; ++tWarpId) {
				mask |= (currentEvent.active[tWarpId] ? 1 : 0);
				mask <<= 1;
			}
			return mask;
		}
		case ir::PTXOperand::tid: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return ( threadId % blockDim.x );
				}
				case ir::PTXOperand::iy: {
					return ( ( threadId / blockDim.x ) % blockDim.y );
				}
				case ir::PTXOperand::iz: {
					return ( ( threadId / ( blockDim.x * blockDim.y ) ) ); 
				}
				case ir::PTXOperand::iw: assertM(false, "%tid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false, "%tid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::ntid: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return blockDim.x;
				}
				case ir::PTXOperand::iy: {
					return blockDim.y;
				}
				case ir::PTXOperand::iz: {
					return blockDim.z; 
				}
				case ir::PTXOperand::iw: assertM(false, "%ntid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false, "%ntid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::warpId: return 0;
		case ir::PTXOperand::warpSize: {
			return blockDim.z * blockDim.y * blockDim.x;
		}
		case ir::PTXOperand::ctaId: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return blockId.x;
				}
				case ir::PTXOperand::iy: {
					return blockId.y;
				}
				case ir::PTXOperand::iz: {
					return blockId.z; 
				}
				case ir::PTXOperand::iw: assertM(false, "%ctaid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false, "%ctaid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::nctaId: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return gridDim.x;
				}
				case ir::PTXOperand::iy: {
					return gridDim.y;
				}
				case ir::PTXOperand::iz: {
					return gridDim.z; 
				}
				case ir::PTXOperand::iw: assertM(false,"%nctaid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false,"%nctaid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::clock: return clock; break;
		default: break;
	}
	
	return 0;
}

ir::PTXF32 executive::CooperativeThreadArray::sat(int modifier, ir::PTXF32 f) {
	if (modifier & ir::PTXInstruction::sat) {
		return (f <= 0 || hydrazine::isnan(f) ? 0 : (f >= 1.0f ? 1.0f : f));
	}
	return f;
}

ir::PTXF64 executive::CooperativeThreadArray::sat(int modifier, ir::PTXF64 f) {
	if (modifier & ir::PTXInstruction::sat) {
		return (f <= 0 || hydrazine::isnan(f) ? 0 : (f >= 1.0 ? 1.0 : f));
	}
	return f;
}

static ir::PTXF32 ftz(int modifier, ir::PTXF32 f) {
	if (modifier & ir::PTXInstruction::ftz) {
		return (!hydrazine::isnormal(f) &&
			!hydrazine::isnan(f) && !hydrazine::isinf(f))
			? hydrazine::copysign(0.0f, f) : f;
	}
	return f;
}

void executive::CooperativeThreadArray::trace() {
	if (traceEvents) {
		currentEvent.contextStackSize =
			(ir::PTXU32)reconvergenceMechanism->stackSize();
		kernel->traceEvent(currentEvent);
	}
}

void executive::CooperativeThreadArray::postTrace() {
	if (traceEvents) {
		currentEvent.contextStackSize =
			(ir::PTXU32)reconvergenceMechanism->stackSize();

		kernel->tracePostEvent(currentEvent);
	}
}

////////////////////////////////////////////////////////////////////////////////

void executive::CooperativeThreadArray::reset() {

	reconvergenceMechanism->initialize();

	barriers.clear();
	barriers.resize(16);	// ir::PTX2.1
	
	size_t threads = getActiveContext().active.size();
	for (BarrierVector::iterator bar_it = barriers.begin();
		bar_it != barriers.end(); ++bar_it) {
		bar_it->initialize(threads);
	}
}

/*! initializes elements of the CTA */
void executive::CooperativeThreadArray::initialize(const ir::Dim3 & block) {

	counter = 0;
	blockId = block;

	currentEvent.blockId  = blockId;
	currentEvent.gridDim  = gridDim;
	currentEvent.blockDim = blockDim;
	
	reset();
}

/*! finishes execution of the CTA */
void executive::CooperativeThreadArray::finalize() {

}

/*!
	Called by the worker thread to evaluate a block
*/
void executive::CooperativeThreadArray::execute(int PC) {
	using namespace ir;

	jumpToPC(PC);
	
	bool running = true;
	assert(reconvergenceMechanism->stackSize());
	
	report("CooperativeThreadArray::execute called");
	report("  block is " << blockId.x << ", "
		<< blockId.y << ", " << blockId.z);
	reportE(REPORT_STATIC_INSTRUCTIONS, "Running " << kernel->toString());

	do {
		assert(reconvergenceMechanism->stackSize());

		// get the context and advance the program counter
		CTAContext& context = getActiveContext();
		const ir::PTXInstruction& instr  = currentInstruction(context);
		const ir::PTXInstruction::Opcode opcode = instr.opcode;

		reconvergenceMechanism->evalPredicate(context);

		reportE(REPORT_DYNAMIC_INSTRUCTIONS, " [PC: " << context.PC 
			<< ", counter: " << counter 
			<< "] " << instr.toString() << " [stack "
			<< reconvergenceMechanism->stackSize() << "] [active " 
			<< context.active.count() << "]" );

		if (traceEvents) {
			currentEvent.reset();
			currentEvent.PC = context.PC;
			currentEvent.instruction = &instr;
			currentEvent.active = context.predicateMask(instr);
		}
		
		switch (instr.opcode) {
			case ir::PTXInstruction::Abs:
				eval_Abs(context, instr); break;
			case ir::PTXInstruction::Add:
				eval_Add(context, instr); break;
			case ir::PTXInstruction::AddC:
				eval_AddC(context, instr); break;
			case ir::PTXInstruction::And:
				eval_And(context, instr); break;
			case ir::PTXInstruction::Atom:
				eval_Atom(context, instr); break;
			case ir::PTXInstruction::Bar:
				eval_Bar(context, instr); break;
			case ir::PTXInstruction::Bfi:
				eval_Bfi(context, instr); break;
			case ir::PTXInstruction::Bfind:
				eval_Bfind(context, instr); break;
			case ir::PTXInstruction::Bfe:
				eval_Bfe(context, instr); break;
			case ir::PTXInstruction::Bra:
				eval_Bra(context, instr); break;
			case ir::PTXInstruction::Brev:
				eval_Brev(context, instr); break;
			case ir::PTXInstruction::Brkpt:
				eval_Brkpt(context, instr); break;
			case ir::PTXInstruction::Call:
				eval_Call(context, instr); break;
			case ir::PTXInstruction::Clz:
				eval_Clz(context, instr); break;
			case ir::PTXInstruction::CNot:
				eval_CNot(context, instr); break;
			case ir::PTXInstruction::CopySign:
				eval_CopySign(context, instr); break;
			case ir::PTXInstruction::Cos:
				eval_Cos(context, instr); break;
			case ir::PTXInstruction::Cvt:
				eval_Cvt(context, instr); break;
			case ir::PTXInstruction::Cvta:
				eval_Cvta(context, instr); break;
			case ir::PTXInstruction::Div:
				eval_Div(context, instr); break;
			case ir::PTXInstruction::Ex2:
				eval_Ex2(context, instr); break;
			case ir::PTXInstruction::Exit:
				eval_Exit(context, instr); break;
			case ir::PTXInstruction::Fma:
				eval_Fma(context, instr); break;
			case ir::PTXInstruction::Isspacep:
				eval_Isspacep(context, instr); break;
			case ir::PTXInstruction::Ld:
				eval_Ld(context, instr); break;
			case ir::PTXInstruction::Lg2:
				eval_Lg2(context, instr); break;
			case ir::PTXInstruction::Ldu:
				eval_Ldu(context, instr); break;
			case ir::PTXInstruction::Mad24:
				eval_Mad24(context, instr); break;
			case ir::PTXInstruction::Mad:
				eval_Mad(context, instr); break;
			case ir::PTXInstruction::Max:
				eval_Max(context, instr); break;
			case ir::PTXInstruction::Membar:
				eval_Membar(context, instr); break;
			case ir::PTXInstruction::Min:
				eval_Min(context, instr); break;
			case ir::PTXInstruction::Mov:
				eval_Mov(context, instr); break;
			case ir::PTXInstruction::Mul24:
				eval_Mul24(context, instr); break;
			case ir::PTXInstruction::Mul:
				eval_Mul(context, instr); break;
			case ir::PTXInstruction::Neg:
				eval_Neg(context, instr); break;
			case ir::PTXInstruction::Not:
				eval_Not(context, instr); break;
			case ir::PTXInstruction::Or:
				eval_Or(context, instr); break;
			case ir::PTXInstruction::Pmevent:
				eval_Pmevent(context, instr); break;
			case ir::PTXInstruction::Popc:
				eval_Popc(context, instr); break;
			case ir::PTXInstruction::Prefetch:
				eval_Prefetch(context, instr); break;
			case ir::PTXInstruction::Prefetchu:
				eval_Prefetchu(context, instr); break;
			case ir::PTXInstruction::Prmt:
				eval_Prmt(context, instr); break;
			case ir::PTXInstruction::Rcp:
				eval_Rcp(context, instr); break;
			case ir::PTXInstruction::Red:
				eval_Red(context, instr); break;
			case ir::PTXInstruction::Rem:
				eval_Rem(context, instr); break;
			case ir::PTXInstruction::Ret:
				eval_Ret(context, instr); break;
			case ir::PTXInstruction::Rsqrt:
				eval_Rsqrt(context, instr); break;
			case ir::PTXInstruction::Sad:
				eval_Sad(context, instr); break;
			case ir::PTXInstruction::SelP:
				eval_SelP(context, instr); break;
			case ir::PTXInstruction::Set:
				eval_Set(context, instr); break;
			case ir::PTXInstruction::SetP:
				eval_SetP(context, instr); break;
			case ir::PTXInstruction::Shr:
				eval_Shr(context, instr); break;
			case ir::PTXInstruction::Shl:
				eval_Shl(context, instr); break;
			case ir::PTXInstruction::Sin:
				eval_Sin(context, instr); break;
			case ir::PTXInstruction::SlCt:
				eval_SlCt(context, instr); break;
			case ir::PTXInstruction::Sqrt:
				eval_Sqrt(context, instr); break;
			case ir::PTXInstruction::St:
				eval_St(context, instr); break;
			case ir::PTXInstruction::Sub:
				eval_Sub(context, instr); break;
			case ir::PTXInstruction::SubC:
				eval_SubC(context, instr); break;
			case ir::PTXInstruction::TestP:
				eval_TestP(context, instr); break;
			case ir::PTXInstruction::Tex:
				eval_Tex(context, instr); break;
			case ir::PTXInstruction::Trap:
				eval_Trap(context, instr); break;
			case ir::PTXInstruction::Vote:
				eval_Vote(context, instr); break;
			case ir::PTXInstruction::Xor:
				eval_Xor(context, instr); break;
			case ir::PTXInstruction::Reconverge:
				eval_Reconverge(context, instr); break;
			default:
				assertM(false, "Opcode at pc " 
					<< context.PC << " - " 
					<< ir::PTXInstruction::toString(instr.opcode) 
					<< " not supported.");
				break;
		}
	
		postTrace();
		
		running = reconvergenceMechanism->nextInstruction(
			getActiveContext(), instr, opcode);

		clock += 4;
		++counter;

	} while (running);

	if(getActiveContext().executionState == CTAContext::Exit)
	{
		finalize();
		report("kernel finished in " << counter << " instructions");
	}
}

void executive::CooperativeThreadArray::jumpToPC(int PC) {
	assert(reconvergenceMechanism->stackSize() != 0);
	assert(PC < (int)kernel->instructions.size());
	
	getActiveContext().PC = PC;
}

int executive::CooperativeThreadArray::getPC() const {
	assert(reconvergenceMechanism->stackSize() != 0);
	
	return getActiveContext().PC;
}

executive::CooperativeThreadArray::RegisterFile 
	executive::CooperativeThreadArray::getCurrentRegisterFile() const {
	RegisterFile file(threadCount * functionCallStack.registerCount());	
	RegisterFile::iterator ri = file.begin();
	for (int thread = 0; thread != threadCount; ++thread) {
		for (unsigned int reg = 0; 
			reg != functionCallStack.registerCount(); ++reg, ++ri) {
			*ri = functionCallStack.registerFilePointer(thread)[reg];
		}
	}
	return file;
}

/*!
	gets the active context of the cooperative thread array
*/
executive::CTAContext & executive::CooperativeThreadArray::getActiveContext() {
	return reconvergenceMechanism->getContext();
}

const executive::CTAContext & executive::CooperativeThreadArray::getActiveContext() const {
	return reconvergenceMechanism->getContext();
}

executive::CTAContext::ExecutionState
	executive::CooperativeThreadArray::getExecutionState() const {
	return reconvergenceMechanism->getContext().executionState;
}

void executive::CooperativeThreadArray::setExecutionState(
	executive::CTAContext::ExecutionState state) {

	reconvergenceMechanism->getContext().executionState = state;
}

////////////////////////////////////////////////////////////////////////////////
//
// Execution support


// Register accessors

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU8 executive::CooperativeThreadArray::getRegAsU8(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU8 r = *( (ir::PTXU8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << (int)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << (int)r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU16 executive::CooperativeThreadArray::getRegAsU16(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU16 r = *( (ir::PTXU16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}
		
/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU32 executive::CooperativeThreadArray::getRegAsU32(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU32 r = *( (ir::PTXU32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU64 executive::CooperativeThreadArray::getRegAsU64(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU64 r = *( (ir::PTXU64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << (void*)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << (void*)r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS8 executive::CooperativeThreadArray::getRegAsS8(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS8 r = *( (ir::PTXS8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << (int)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << (int)r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS16 executive::CooperativeThreadArray::getRegAsS16(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS16 r = *( (ir::PTXS16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}
		
/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS32 executive::CooperativeThreadArray::getRegAsS32(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS32 r = *( (ir::PTXS32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS64 executive::CooperativeThreadArray::getRegAsS64(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS64 r = *( (ir::PTXS64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXF32 executive::CooperativeThreadArray::getRegAsF32(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXF32 r = *( (ir::PTXF32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU32>(r))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(r))
			<< std::dec << ")");
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXF64 executive::CooperativeThreadArray::getRegAsF64(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXF64 r = *( (ir::PTXF64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(r))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(r))
			<< std::dec << ")");
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB8 executive::CooperativeThreadArray::getRegAsB8(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB8 r = *( (ir::PTXB8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << (int)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << (int)r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB16 executive::CooperativeThreadArray::getRegAsB16(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB16 r = *( (ir::PTXB16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}
		
/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB32 executive::CooperativeThreadArray::getRegAsB32(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB32 r = *( (ir::PTXB32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB64 executive::CooperativeThreadArray::getRegAsB64(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB64 r = *( (ir::PTXB64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << (void*)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID 
		<< " reg " << reg << " <= " << (void*)r);
	#endif
	return r;
}

/*!
	Gets a register value
	
	\param threadID ID of the active thread
	\param reg index of register
*/
bool executive::CooperativeThreadArray::getRegAsPredicate(int threadID, 
	ir::PTXOperand::RegisterType reg) {
	bool r = (bool)getRegAsU64(threadID, reg);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID 
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS && REPORT_PREDICATE_READS,
		"   thread " << threadID 
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU8(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXU8 value) {
	ir::PTXU8* r = (ir::PTXU8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << (int)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << (int)value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU16(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXU16 value) {
	ir::PTXU16* r = (ir::PTXU16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}
		
/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU32(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXU32 value) {
	ir::PTXU32* r = (ir::PTXU32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU64(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXU64 value) {
	ir::PTXU64* r = (ir::PTXU64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << (void*)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << (void*)value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS8(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXS8 value) {
	ir::PTXS8* r = (ir::PTXS8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << (int)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << (int)value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS16(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXS16 value) {
	ir::PTXS16* r = (ir::PTXS16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}
		
/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS32(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXS32 value) {
	ir::PTXS32* r = (ir::PTXS32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS64(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXS64 value) {
	ir::PTXS64* r = (ir::PTXS64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsF32(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXF32 value) {
	ir::PTXF32* r = (ir::PTXF32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU32>(value))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU32>(value))
			<< std::dec << ")");
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsF64(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXF64 value) {
	ir::PTXF64* r = (ir::PTXF64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(value))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(value))
			<< std::dec << ")");
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB8(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXB8 value) {
	ir::PTXB8* r = (ir::PTXB8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << (int)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << (int)value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB16(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXB16 value) {
	ir::PTXB16* r = (ir::PTXB16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}
		
/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB32(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXB32 value) {
	ir::PTXB32* r = (ir::PTXB32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value 

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB64(int threadID, 
	ir::PTXOperand::RegisterType reg, ir::PTXB64 value) {
	ir::PTXB64* r = (ir::PTXB64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value
	
	\param threadID ID of the active thread
	\param reg index of register
	\param value value of predicate register
*/
void executive::CooperativeThreadArray::setRegAsPredicate(int threadID, 
	ir::PTXOperand::RegisterType reg, bool value) {
	ir::PTXB64* r = (ir::PTXB64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID 
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES,
		"   thread " << threadID 
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
ir::PTXU8 executive::CooperativeThreadArray::operandAsU8(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU8(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU8(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXU8)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXU8)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXU16 executive::CooperativeThreadArray::operandAsU16(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU16(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU16(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXU16)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXU16)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXU32 executive::CooperativeThreadArray::operandAsU32(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU32(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU32(threadID, op.reg);
		case ir::PTXOperand::Address:
			return (ir::PTXU32)(op.imm_uint) + op.offset;
		case ir::PTXOperand::Immediate:
			return (ir::PTXU32)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXU32)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXU64 executive::CooperativeThreadArray::operandAsU64(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU64(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU64(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXU64)(op.imm_uint);
		case ir::PTXOperand::Address:
			return (ir::PTXU64)(op.imm_uint) + op.offset;
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS8 executive::CooperativeThreadArray::operandAsS8(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS8(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS8(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXS8)(op.imm_int);
		case ir::PTXOperand::Special:
			return (ir::PTXS8)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS16 executive::CooperativeThreadArray::operandAsS16(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS16(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS16(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXS16)(op.imm_int);
		case ir::PTXOperand::Special:
			return (ir::PTXS16)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS32 executive::CooperativeThreadArray::operandAsS32(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS32(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS32(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXS32)(op.imm_int);
		case ir::PTXOperand::Special:
			return (ir::PTXS32)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS64 executive::CooperativeThreadArray::operandAsS64(int threadID, 
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS64(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS64(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXS64)(op.imm_int);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXF32 executive::CooperativeThreadArray::operandAsF32(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Register:
			return getRegAsF32(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXF32)(op.imm_single);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXF64 executive::CooperativeThreadArray::operandAsF64(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Register:
			return getRegAsF64(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXF64)(op.imm_float);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB8 executive::CooperativeThreadArray::operandAsB8(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB8(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB8(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB8)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXB8)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB16 executive::CooperativeThreadArray::operandAsB16(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB16(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB16(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB16)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXB16)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB32 executive::CooperativeThreadArray::operandAsB32(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB32(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB32(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB32)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXB32)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB64 executive::CooperativeThreadArray::operandAsB64(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB64(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB64(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB64)(op.imm_uint);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

bool executive::CooperativeThreadArray::operandAsPredicate(int threadID,
	const ir::PTXOperand &op) {
	bool result = false;
	switch (op.addressMode) {
		case ir::PTXOperand::Register:
			result = getRegAsPredicate(threadID, op.reg);
			break;
		case ir::PTXOperand::Immediate:
			return (bool)(op.imm_uint);
		default:
			assert(0 == "invalid address mode of operand");
	}
	
	switch (op.condition) {
		case ir::PTXOperand::PT:
			result = true;
			break;
		case ir::PTXOperand::nPT:
			result = false;
			break;
		case ir::PTXOperand::Pred:
			break;
		case ir::PTXOperand::InvPred:
			result = !result;
			break;	
	}
	
	return result;
}

template<typename T>
T executive::CooperativeThreadArray::getFunctionParameter(int threadID,
	const ir::PTXInstruction& i, int index)
{
	assert((size_t)index < i.b.array.size());

	T value;

	char* base    = (char*) functionCallStack.stackFramePointer(threadID);
	void* address = base + i.b.array[index].offset;

	std::memcpy(&value, address, sizeof(T));

	return value;
}

template<typename T>
void executive::CooperativeThreadArray::setFunctionParameter(int threadID,
	const ir::PTXInstruction& i, int index, T value)
{
	assert((size_t)index < i.d.array.size());
	
	char* base    = (char*) functionCallStack.stackFramePointer(threadID);
	void* address = base + i.d.array[index].offset;
	
	std::memcpy(address, &value, sizeof(T));
}
////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Abs(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {

			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = ftz(instr.modifier, CTAAbs(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s8) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS8 d, a = operandAsS8(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS8(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Add(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)), 
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			d = ftz(instr.modifier, sat(instr.modifier, a + b));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a), 
				b = operandAsF64(threadID, instr.b);
			d = a + b;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a), 
				b = operandAsS16(threadID, instr.b);
			d = a + b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a), 
				b = operandAsS32(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::sat) {
				ir::PTXS64 temp = (ir::PTXS64)a + (ir::PTXS64)b;
				temp = min(temp, (ir::PTXS64)INT_MAX);
				temp = max(temp, (ir::PTXS64)INT_MIN);
				d = (ir::PTXS32)temp;
			}
			else if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXS32 carry = 0;
				hydrazine::add(d, carry, a, b, carry);
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a + b;
			}
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a), 
				b = operandAsS64(threadID, instr.b);
			d = a + b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a), 
				b = operandAsU16(threadID, instr.b);
			d = a + b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a), 
				b = operandAsU32(threadID, instr.b);
			if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXU32 carry = 0;
				hydrazine::add(d, carry, a, b, carry);
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a + b;
			}
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a), 
				b = operandAsU64(threadID, instr.b);
			d = a + b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_AddC(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {

	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d = 0,
				a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);

			ir::PTXU32 carry = getRegAsU32(threadID, instr.c.reg);
			hydrazine::add(d, carry, a, b, carry);

			setRegAsU32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d = 0,
				a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);

			ir::PTXS32 carry = getRegAsS32(threadID, instr.c.reg);
			hydrazine::add(d, carry, a, b, carry);

			setRegAsS32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsS32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_And(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			bool d, a = operandAsPredicate(threadID, instr.a), 
				b = operandAsPredicate(threadID, instr.b);
			d = b && a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a), 
				b = operandAsB16(threadID, instr.b);
			d = a & b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a), 
				b = operandAsB32(threadID, instr.b);
			d = a & b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 d, a = operandAsB64(threadID, instr.a), 
				b = operandAsB64(threadID, instr.b);
			d = a & b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Atom(CTAContext &context, const ir::PTXInstruction &instr) {
	size_t elementSize = 0;
	
	switch (instr.type) {
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			const char *source = 0;

			switch (instr.a.addressMode) {
				case ir::PTXOperand::Indirect:
					source += instr.a.offset;
				case ir::PTXOperand::Register:
					source += getRegAsU64(threadID, instr.a.reg);				
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.a.imm_uint;
					source += instr.a.offset;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand", 
						context.PC, instr);
			}

			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;

		const char *source = 0;

		switch (instr.a.addressMode) {
			case ir::PTXOperand::Indirect:
				source += instr.a.offset;
			case ir::PTXOperand::Register:
				source += getRegAsU64(threadID, instr.a.reg);				
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.a.imm_uint;
				source += instr.a.offset;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand", 
					context.PC, instr);
		}

		switch (instr.addressSpace) {
			case ir::PTXInstruction::Global:
				{	

				}
				break;
			case ir::PTXInstruction::Shared:
				{
					source += (ir::PTXU64) functionCallStack.sharedMemoryPointer();
				}
				break;
			default:
				throw RuntimeException("unsupported address space", 
					context.PC, instr);
		}
		
		switch (instr.atomicOperation) {
			case ir::PTXInstruction::AtomicAnd:
				{
					if(instr.type != ir::PTXOperand::b32
						&& instr.type != ir::PTXOperand::s32
						&& instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}
					ir::PTXB32 d = *((ir::PTXB32*)source);
					setRegAsB32(threadID, instr.d.reg, d);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					*((ir::PTXB32*)source) = d & b;
					reportE(REPORT_ATOM, "Atomically updated " << d << " to " 
						<< *((ir::PTXB32*)source) );		
				}
				break;
			case ir::PTXInstruction::AtomicOr:
				{
					if(instr.type != ir::PTXOperand::b32
						&& instr.type != ir::PTXOperand::s32
						&& instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}
					ir::PTXB32 d = *((ir::PTXB32*)source);
					setRegAsB32(threadID, instr.d.reg, d);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					*((ir::PTXB32*)source) = d | b;
					reportE(REPORT_ATOM, "Atomically updated " << d << " to " 
						<< *((ir::PTXB32*)source) );		
				}
				break;
			case ir::PTXInstruction::AtomicXor:
				{
					if(instr.type != ir::PTXOperand::b32
						&& instr.type != ir::PTXOperand::s32
						&& instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}
					ir::PTXB32 d = *((ir::PTXB32*)source);
					setRegAsB32(threadID, instr.d.reg, d);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					*((ir::PTXB32*)source) = d ^ b;				
					reportE(REPORT_ATOM, "Atomically updated " << d << " to " 
						<< *((ir::PTXB32*)source) );		
				}
				break;
			case ir::PTXInstruction::AtomicCas:
				{
					if (instr.type == ir::PTXOperand::b32
						|| instr.type == ir::PTXOperand::s32
						|| instr.type == ir::PTXOperand::u32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						setRegAsB32(threadID, instr.d.reg, d);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						ir::PTXB32 c = operandAsB32(threadID, instr.c);
						*((ir::PTXB32*)source) = (d==b) ? c : d;				
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXB32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::b64
						|| instr.type == ir::PTXOperand::s64
						|| instr.type == ir::PTXOperand::u64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						setRegAsB64(threadID, instr.d.reg, d);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						ir::PTXB64 c = operandAsB64(threadID, instr.c);
						*((ir::PTXB64*)source) = (d==b) ? c : d;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXB64*)source) );		
					}
					else {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}				
				}
				break;
			case ir::PTXInstruction::AtomicExch:
				{
					if (instr.type == ir::PTXOperand::b32
						|| instr.type == ir::PTXOperand::s32
						|| instr.type == ir::PTXOperand::u32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						setRegAsB32(threadID, instr.d.reg, d);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						*((ir::PTXB32*)source) = b;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXB32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::b64
						|| instr.type == ir::PTXOperand::s64
						|| instr.type == ir::PTXOperand::u64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						setRegAsB64(threadID, instr.d.reg, d);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						*((ir::PTXB64*)source) = b;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXB64*)source) );		
					}
					else {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}				
				}
				break;
			case ir::PTXInstruction::AtomicAdd:
				{
					if (instr.type == ir::PTXOperand::u32) {
						ir::PTXU32 d = *((ir::PTXU32*)source);
						setRegAsU32(threadID, instr.d.reg, d);
						ir::PTXU32 b = operandAsU32(threadID, instr.b);
						*((ir::PTXU32*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXU32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::s32) {
						ir::PTXS32 d = *((ir::PTXS32*)source);
						setRegAsS32(threadID, instr.d.reg, d);
						ir::PTXS32 b = operandAsS32(threadID, instr.b);
						*((ir::PTXS32*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXS32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::f32) {
						ir::PTXF32 d = *((ir::PTXF32*)source);
						setRegAsF32(threadID, instr.d.reg, d);
						ir::PTXF32 b = operandAsF32(threadID, instr.b);
						*((ir::PTXF32*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXF32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::u64) {
						ir::PTXU64 d = *((ir::PTXU64*)source);
						setRegAsU64(threadID, instr.d.reg, d);
						ir::PTXU64 b = operandAsU64(threadID, instr.b);
						*((ir::PTXU64*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXU64*)source) );		
					}
					else {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}				
				}
				break;
			case ir::PTXInstruction::AtomicInc:
				{
					if(instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}
					ir::PTXU32 d = *((ir::PTXU32*)source);
					setRegAsU32(threadID, instr.d.reg, d);
					ir::PTXU32 b = operandAsU32(threadID, instr.b);
					*((ir::PTXU32*)source) = (d >= b) ? 0 : d + 1;
					reportE(REPORT_ATOM, "Atomically updated " << d 
						<< " to " << *((ir::PTXU32*)source) );		
				}
				break;
			case ir::PTXInstruction::AtomicDec:
				{
					if(instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}
					ir::PTXU32 d = *((ir::PTXU32*)source);
					setRegAsU32(threadID, instr.d.reg, d);
					ir::PTXU32 b = operandAsU32(threadID, instr.b);
					*((ir::PTXU32*)source) = ((d == 0) || (d > b)) ? b : d - 1;				
					reportE(REPORT_ATOM, "Atomically updated " << d 
						<< " to " << *((ir::PTXU32*)source) );		
				}
				break; 
			case ir::PTXInstruction::AtomicMin:
				{
					if (instr.type == ir::PTXOperand::u32) {
						ir::PTXU32 d = *((ir::PTXU32*)source);
						setRegAsU32(threadID, instr.d.reg, d);
						ir::PTXU32 b = operandAsU32(threadID, instr.b);
						*((ir::PTXU32*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXU32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::s32) {
						ir::PTXS32 d = *((ir::PTXS32*)source);
						setRegAsS32(threadID, instr.d.reg, d);
						ir::PTXS32 b = operandAsS32(threadID, instr.b);
						*((ir::PTXS32*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXS32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::f32) {
						ir::PTXF32 d = *((ir::PTXF32*)source);
						setRegAsF32(threadID, instr.d.reg, d);
						ir::PTXF32 b = operandAsF32(threadID, instr.b);
						*((ir::PTXF32*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXF32*)source) );		
					}
					else {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}				
				}
				break;
			case ir::PTXInstruction::AtomicMax:
				{
					if (instr.type == ir::PTXOperand::u32) {
						ir::PTXU32 d = *((ir::PTXU32*)source);
						setRegAsU32(threadID, instr.d.reg, d);
						ir::PTXU32 b = operandAsU32(threadID, instr.b);
						*((ir::PTXU32*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXU32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::s32) {
						ir::PTXS32 d = *((ir::PTXS32*)source);
						setRegAsS32(threadID, instr.d.reg, d);
						ir::PTXS32 b = operandAsS32(threadID, instr.b);
						*((ir::PTXS32*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXS32*)source) );		
					}
					else if (instr.type == ir::PTXOperand::f32) {
						ir::PTXF32 d = *((ir::PTXF32*)source);
						setRegAsF32(threadID, instr.d.reg, d);
						ir::PTXF32 b = operandAsF32(threadID, instr.b);
						*((ir::PTXF32*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d 
							<< " to " << *((ir::PTXF32*)source) );		
					}
					else {
						throw RuntimeException("invalid data type", 
							context.PC, instr);						
					}				
				}
				break;
			default:
				throw RuntimeException("Invalid atomic operation", 
					context.PC, instr);
		}
	}
}

void executive::CooperativeThreadArray::eval_Bfi(CTAContext &context, const 
	ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::b32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB32 pq = operandAsB32(threadID, instr.pq);
			ir::PTXB32 a = operandAsB32(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU32 c = operandAsU32(threadID, instr.c);
			ir::PTXB32 d = hydrazine::bitFieldInsert(pq, a, b, c);
			setRegAsB32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::b64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 pq = operandAsB64(threadID, instr.pq);
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU32 c = operandAsU32(threadID, instr.c);
			ir::PTXB64 d = hydrazine::bitFieldInsert(pq, a, b, c);
			setRegAsB64(threadID, instr.d.reg, d);
		}
		break;
	}
	default: {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Bfind(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::u32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a);
			d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::s32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 a = operandAsS32(threadID, instr.a);
			ir::PTXU32 d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::s64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 a = operandAsS64(threadID, instr.a);
			ir::PTXU32 d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::u64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 a = operandAsU64(threadID, instr.a);
			ir::PTXU32 d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	default: {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
	}
}

void executive::CooperativeThreadArray::eval_Brev(CTAContext& context, 
	const ir::PTXInstruction& instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::b32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB32 a = operandAsB32(threadID, instr.a);
			ir::PTXB32 d = hydrazine::brev(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::b64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXB64 d = hydrazine::brev(a);
			setRegAsB64(threadID, instr.d.reg, d);
		}
		break;
	}
	default: {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
	}
}

void executive::CooperativeThreadArray::eval_Bfe(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int tid = 0; tid < threadCount; tid++) {
		if (!context.predicated(tid, instr)) {
			continue;
		}
		bool size32bit = (instr.type == ir::PTXOperand::u32
			|| instr.type == ir::PTXOperand::s32);
		bool isSigned = (instr.type == ir::PTXOperand::s32
			|| instr.type == ir::PTXOperand::s64);
		
		ir::PTXU32 msb    = (size32bit ? 31 : 63);
		ir::PTXU32 pos    = operandAsU32(tid, instr.b);
		ir::PTXU32 len    = operandAsU32(tid, instr.c);
		ir::PTXU64 a      = operandAsU64(tid, instr.a);
		ir::PTXU64 mask   = ((1 << len) - 1);
		ir::PTXU64 result = 0;
		
		if (isSigned) {
			result = (msb ? -1 : 0) & (~mask);
		}
		result |= ((a >> pos) & mask);
		if (size32bit) {
			setRegAsU32(tid, instr.d.reg,
				hydrazine::bit_cast<ir::PTXU32, ir::PTXU64>(result));
		}
		else {
			setRegAsU64(tid, instr.d.reg, result);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Bar(CTAContext& context, 
	const ir::PTXInstruction& instr) {
	trace();
	
	size_t participating = context.active.size();
	
	if (instr.a.addressMode != ir::PTXOperand::Invalid) {
		report("eval_Bar() - barrier name must be constant");
		throw RuntimeException("barrier name must be a constant in this "
			"version of Ocelot ir::PTX Emulator",
			context.PC, instr);
	}
	
	if (instr.b.addressMode != ir::PTXOperand::Invalid) {
		if (instr.b.addressMode == ir::PTXOperand::Immediate) {
			participating = (size_t)instr.b.imm_uint;
		}
		else {
			report("eval_Bar() - number of threads participating in barrier "
				"must be constant");
			throw RuntimeException(
				"number of threads participating in barrier must be "
					"constant operand in this version of Ocelot",
				context.PC, instr);
		}
	}
	
	if (participating != context.active.size()) {
		throw RuntimeException(
			"Ocelot does not yet support partial-CTA barriers",
			context.PC, instr);
	}
	
	reconvergenceMechanism->eval_Bar(context, instr);
}

void executive::CooperativeThreadArray::eval_Bra(CTAContext &context,
	const ir::PTXInstruction &instr) {
	using namespace boost;

	dynamic_bitset<> branch = context.active;
	dynamic_bitset<> fall_through(branch.size(), 0);

	// determine divergence
	for (int i = 0; i < threadCount; i++) {
		if (context.predicated(i, instr)) {
			// typical branch
			branch[i] = context.active[i];
			fall_through[i] = false;
		}
		else {
			fall_through[i] = context.active[i];
			branch[i] = false;
		}
	}

	if (traceEvents) {
		currentEvent.fallthrough = fall_through;
		currentEvent.taken = branch;
	}

#if REPORT_BRA
	report("  active threads       [" << context.active.count() << "] "
		<< context.active);
	report("  branching threads    [" << branch.count() << "] " << branch);
	report("  fall-through threads [" << fall_through.count() << "] "
		<< fall_through);
	report("  branch target PC " << instr.branchTargetInstruction);
	report("  reconverge PC " << instr.reconvergeInstruction);
#endif

	// dispatch to reconvergence mechanism
	reconvergenceMechanism->eval_Bra(context, instr, branch, fall_through);

	trace();
}

/*!
	Reconverge instruction is inserted into the ir::PTX during analysis and construction of the
	emulated kernel. 
*/
void executive::CooperativeThreadArray::eval_Reconverge(
	CTAContext &context, const ir::PTXInstruction &instr) {
	using namespace std;
	trace();
	
	// dispatch
	reconvergenceMechanism->eval_Reconverge(context, instr);
}

/*!

*/
void executive::CooperativeThreadArray::eval_Brkpt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	context.executionState = CTAContext::Break;
}

void executive::CooperativeThreadArray::copyArgument(const ir::PTXOperand& s,
	CTAContext& context) {
	reportE(REPORT_CALL, " Copying " << ir::PTXOperand::bytes(s.type) 
		<< " bytes from previous stack frame at " << s.offset
		<< " to current frame at " << s.offset );
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.active[thread]) continue;
		char* stackPointer = (char*)functionCallStack.stackFramePointer(thread);
		char* previousStackPointer =
			(char*)functionCallStack.previousStackFramePointer(thread);
		std::memcpy(stackPointer + s.offset, previousStackPointer + s.offset,
			ir::PTXOperand::bytes(s.type));
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Call(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	typedef std::unordered_map<ir::PTXU64, CTAContext> TargetMap;
	trace();
	
	// copy any register operands into parameter stack memory
	for (ir::PTXOperand::Array::const_iterator 
		argument = instr.b.array.begin();
		argument != instr.b.array.end(); ++argument) {
		if (argument->addressMode != ir::PTXOperand::Register) continue;
		
		for (int threadID = 0; threadID != threadCount; ++threadID) {
			if(!context.predicated(threadID, instr)) continue;

			ir::PTXU64 data = getRegAsU64(threadID, argument->reg);
			
			char* base = (char*) functionCallStack.stackFramePointer(threadID);
				
			std::memcpy(base + argument->offset, &data,
				ir::PTXOperand::bytes(argument->type));
		}
	}
	
	// Is this a direct or indirect call?
	if (instr.a.addressMode == ir::PTXOperand::Register) {
		// Complex indirect call handling
		reportE(REPORT_CALL, " indirect call [" << context.active << "]" );
		
		// Create contexts for all unique targets
		TargetMap targets;
		
		for (int threadID = 0; threadID != threadCount; ++threadID) {
			if(!context.predicated(threadID, instr)) continue;
			ir::PTXU64 targetPC = getRegAsU64(threadID, instr.a.reg);
			reportE(REPORT_CALL, "  Thread " << threadID 
				<< " jumped to PC " << targetPC )
			TargetMap::iterator targetContext = targets.find(targetPC);
			if (targetContext == targets.end()) {
				CTAContext newContext(context);
				newContext.active.reset();
				newContext.PC = targetPC;
				targetContext = targets.insert(
					std::make_pair(targetPC, newContext)).first;
			}
			
			targetContext->second.active[threadID] = true;
		}
		
		reportE(REPORT_CALL, " Created " << targets.size() 
			<< " unique target contexts.");
		
		// Get a reference to the base stack pointer
		unsigned int stackSize = functionCallStack.stackFrameSize();
		unsigned int stackOffset = functionCallStack.offset();
		unsigned int callPC = context.PC;
		functionCallStack.saveFrame();

		++context.PC;
		
		reportE(REPORT_CALL, " Caller stack starts at offset " << stackOffset);
		reportE(REPORT_CALL, " Caller size " << stackSize);
		
		// Push all active contexts
		for (TargetMap::iterator targetContext = targets.begin(); 
			targetContext != targets.end(); ++targetContext) {

			const EmulatedKernel* targetKernel = kernel->getKernel(
				targetContext->second.PC);
			if( targetKernel == 0 ) {
				std::stringstream stream;

				stream << targetContext->second.PC;

				throw RuntimeException("attempted to call function at PC '"
					+ stream.str() + "' in module '"
					+ kernel->module->path() +
					"', but there is no function at that PC.",
					context.PC, instr);
			}

			reportE(REPORT_CALL, 
				"  call was taken, increasing stack size by (" 
				<< targetKernel->parameterMemorySize() << " parameter) (" 
				<< targetKernel->registerCount() << " registers) (" 
				<< targetKernel->localMemorySize() << " local memory) (" 
				<< targetKernel->totalSharedMemorySize() 
				<< " sharedMemorySize)");

			functionCallStack.pushFrame(targetKernel->parameterMemorySize(), 
				targetKernel->registerCount(), targetKernel->localMemorySize(), 
				targetKernel->totalSharedMemorySize(), callPC, 
				stackOffset, stackSize);
			for (ir::PTXOperand::Array::const_iterator 
				argument = instr.b.array.begin();
				argument != instr.b.array.end(); ++argument) {
				for (int threadID = 0; threadID != threadCount; ++threadID) {
					if(!targetContext->second.active[threadID]) continue;
					char* base = (char*)
						functionCallStack.savedStackFramePointer(threadID);
					char* target = (char*)functionCallStack.stackFramePointer(
						threadID);
					std::memcpy(target + argument->offset,
						base + argument->offset, 
						ir::PTXOperand::bytes(argument->type));
				}
			}

			reconvergenceMechanism->push(targetContext->second);	
		}
	}
	else {

		// was this function external?
		ir::ExternalFunctionSet::ExternalFunction*
			external = kernel->findExternalFunction(instr.a.identifier);
		if(external != 0) {
			// get the prototype
			ir::Module::FunctionPrototypeMap::const_iterator prototype
				= kernel->module->prototypes().find(instr.a.identifier);
			if(prototype == kernel->module->prototypes().end()) {
				throw RuntimeException("no prototype for external function '"
					+ instr.a.identifier + "' in module '"
					+ kernel->module->path() + "'", context.PC, instr);
			}
			
			for (int threadID = 0; threadID != threadCount; ++threadID) {
				if(!context.predicated(threadID, instr)) continue;
				reportE(REPORT_CALL, " thread " << threadID
					<< " calling external function " << instr.a.identifier);			
				external->call(functionCallStack.stackFramePointer(threadID),
					prototype->second);
			}
			
			// copy register operands back from the stack
			for (ir::PTXOperand::Array::const_iterator 
				argument = instr.d.array.begin();
				argument != instr.d.array.end(); ++argument) {
				if (argument->addressMode != ir::PTXOperand::Register) continue;
		
				for (int threadID = 0; threadID != threadCount; ++threadID) {
					if(!context.predicated(threadID, instr)) continue;

					ir::PTXU64 data = 0;
					char* base =
						(char*) functionCallStack.stackFramePointer(threadID);
				
					std::memcpy(&data, base + argument->offset,
						ir::PTXOperand::bytes(argument->type));
					setRegAsU64(threadID, argument->reg, data);
				}
			}
			
			++context.PC;
		}
		else
		{
			// Handle Cuda Nested Parallelism
			if (instr.a.identifier == "cudaLaunchDevice") {
				eval_cudaLaunchDevice(context, instr);
				return;
			}
			if (instr.a.identifier == "cudaDeviceSynchronize") {
				eval_cudaSynchronizeDevice(context, instr);
				return;
			}

			reportE(REPORT_CALL, " direct call to PC " 
				<< instr.branchTargetInstruction)

			// Handle lazy function linking
			if (instr.branchTargetInstruction == -1) {
				reportE(REPORT_CALL, " lazy linking against kernel '" 
					<< instr.a.identifier << "'");
				kernel->lazyLink(context.PC, instr.a.identifier);
				currentEvent.instruction = &currentInstruction(context);
			}

			const ir::PTXInstruction& jittedInstr = currentInstruction(context);
			assert(jittedInstr.branchTargetInstruction != -1);

			// Simple direct call handling
			if (jittedInstr.uni) {
				reportE(REPORT_CALL, " uniform direct call" );
				int firstActive = context.active.find_first();
				bool taken = context.predicated(firstActive, jittedInstr);		
			
				if (taken) {
					reportE(REPORT_CALL, 
						"  call was taken, increasing stack size by (" 
						<< jittedInstr.a.stackMemorySize << " stack) (" 
						<< jittedInstr.a.registerCount << " registers) (" 
						<< jittedInstr.a.localMemorySize << " local memory) (" 
						<< jittedInstr.a.sharedMemorySize
						<< " sharedMemorySize)");
					functionCallStack.pushFrame(jittedInstr.a.stackMemorySize, 
						jittedInstr.a.registerCount,
						jittedInstr.a.localMemorySize, 
						jittedInstr.a.sharedMemorySize, context.PC, 
						functionCallStack.offset(),
						functionCallStack.stackFrameSize());
					for (ir::PTXOperand::Array::const_iterator 
						argument = jittedInstr.b.array.begin();
						argument != jittedInstr.b.array.end(); ++argument) {
						copyArgument(*argument, context);
					}
				
					CTAContext targetContext(context);
				
					targetContext.PC = jittedInstr.branchTargetInstruction;
					++context.PC;
				
					reconvergenceMechanism->push(targetContext);
				}
			}
			else {
				reportE(REPORT_CALL, " divergent direct call" );

				CTAContext targetContext(context);

				targetContext.active = context.predicateMask(jittedInstr);
				
				reportE(REPORT_CALL, " taken threads - ["
					<< targetContext.active << "]" );
		
				if (targetContext.active.any()) {
					reportE(REPORT_CALL, 
						"  call was taken, increasing stack size by (" 
						<< jittedInstr.a.stackMemorySize << " stack) (" 
						<< jittedInstr.a.registerCount << " registers) (" 
						<< jittedInstr.a.localMemorySize << " local memory) (" 
						<< jittedInstr.a.sharedMemorySize
						<< " sharedMemorySize)");
					functionCallStack.pushFrame(jittedInstr.a.stackMemorySize, 
						jittedInstr.a.registerCount,
						jittedInstr.a.localMemorySize, 
						jittedInstr.a.sharedMemorySize, context.PC, 
						functionCallStack.offset(),
						functionCallStack.stackFrameSize());
					for (ir::PTXOperand::Array::const_iterator 
						argument = jittedInstr.b.array.begin();
						argument != jittedInstr.b.array.end(); ++argument) {
						copyArgument(*argument, targetContext);
					}
				
					targetContext.PC = jittedInstr.branchTargetInstruction;
				
					++context.PC;
					reconvergenceMechanism->push(targetContext);
				}
				else {
					++context.PC;
				}
			}
		}
	}
}


/*!

*/
void executive::CooperativeThreadArray::eval_Clz(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = hydrazine::countLeadingZeros(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXB32 d = hydrazine::countLeadingZeros(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_CNot(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a);
			d = (a == 0 ? 1 : 0);
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = (a == 0 ? 1 : 0);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB64 d, a = operandAsB64(threadID, instr.a);
			d = (a == 0 ? 1 : 0);
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_CopySign(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			
			ir::PTXF32 a = operandAsF32(tid, instr.a);
			ir::PTXF32 b = operandAsF32(tid, instr.b);
			ir::PTXU32 d = (hydrazine::bit_cast<ir::PTXU32>(b) & 0x7fffffff) | 
				(hydrazine::bit_cast<ir::PTXU32>(a) & 0x80000000);

			setRegAsF32(tid, instr.d.reg,
				hydrazine::bit_cast<ir::PTXF32>(d));
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			
			ir::PTXF64 a = operandAsF64(tid, instr.a);
			ir::PTXF64 b = operandAsF64(tid, instr.b);
			ir::PTXU64 d = (hydrazine::bit_cast<ir::PTXU64>(b)
				& 0x7fffffffffffffffULL) | 
				(hydrazine::bit_cast<ir::PTXU64>(a) 
				& 0x8000000000000000ULL);

			setRegAsF64(tid, instr.d.reg, 
				hydrazine::bit_cast<ir::PTXF64>(d));
		}	
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Cos(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = (ir::PTXF32)cos(a);
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = cos(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

template< typename Int >
static ir::PTXF32 toF32(Int value, int modifier) {
	int mode = hydrazine::fegetround();
	if (modifier & ir::PTXInstruction::rn) {
		hydrazine::fesetround(FE_TONEAREST);
	} else if (modifier & ir::PTXInstruction::rz) {
		hydrazine::fesetround(FE_TOWARDZERO);
	} else if (modifier & ir::PTXInstruction::rm) {
		hydrazine::fesetround(FE_DOWNWARD);
	} else if (modifier & ir::PTXInstruction::rp) {
		hydrazine::fesetround(FE_UPWARD);
	}
	ir::PTXF32 d = value;
	hydrazine::fesetround(mode);
	return d;
}

template< typename Int >
static ir::PTXF64 toF64(Int value, int modifier) {
	int mode = hydrazine::fegetround();
	if (modifier & ir::PTXInstruction::rn) {
		hydrazine::fesetround(FE_TONEAREST);
	} else if (modifier & ir::PTXInstruction::rz) {
		hydrazine::fesetround(FE_TOWARDZERO);
	} else if (modifier & ir::PTXInstruction::rm) {
		hydrazine::fesetround(FE_DOWNWARD);
	} else if (modifier & ir::PTXInstruction::rp) {
		hydrazine::fesetround(FE_UPWARD);
	}
	ir::PTXF64 d = value;
	hydrazine::fesetround(mode);
	return d;
}

template< typename Float >
static Float roundToInt(Float a, int modifier, executive::CTAContext &context, 
	const ir::PTXInstruction &instr) {
	Float fd = 0;
	if (modifier & ir::PTXInstruction::rni) {
		fd = hydrazine::nearbyintf(a);
	} else if (modifier & ir::PTXInstruction::rzi) {
		fd = hydrazine::trunc(a);
	} else if (modifier & ir::PTXInstruction::rmi) {
		fd = floor(a);
	} else if (modifier & ir::PTXInstruction::rpi) {
		fd = ceil(a);
	}
	else {
		fd = a;
	}
	return fd;
}

/*!

*/
void executive::CooperativeThreadArray::eval_Cvt(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		
		ir::PTXOperand::DataType sourceType = instr.a.type;
		
		if (instr.a.relaxedType != ir::PTXOperand::TypeSpecifier_invalid) {
			sourceType = instr.a.relaxedType;
		}
		
		switch (sourceType) {
			case ir::PTXOperand::b8: // fall through
			case ir::PTXOperand::u8:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8: // fall through
					case ir::PTXOperand::s16: // fall through
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::s64: // fall through
					case ir::PTXOperand::u64: // fall through
						{
							setRegAsU64(threadID, instr.d.reg, 
								operandAsB8(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXU8 a = operandAsU8(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, CHAR_MAX);
							}
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsB8(threadID, instr.a), 
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsB8(threadID, instr.a), 
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s8:
			{
				switch (instr.type) {
					case ir::PTXOperand::s8: // fall through
					case ir::PTXOperand::s16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::s64:
						{
							ir::PTXS8 a = operandAsS8(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::u8: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							ir::PTXS8 a = operandAsS8(threadID, instr.a);
							if (instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							setRegAsU64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsS8(threadID, instr.a), 
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsS8(threadID, instr.a), 
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::b16: // fall through
			case ir::PTXOperand::u16:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU16 a = operandAsU16(threadID, instr.a);
							ir::PTXU8 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::s64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg, 
								operandAsB16(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXU16 a = operandAsU16(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, CHAR_MAX);
							}
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXU16 a = operandAsU16(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, SHRT_MAX);
							}
							ir::PTXS16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsB16(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsB16(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s16:
			{
				// s16 to one of the following
				switch (instr.type) {
					case ir::PTXOperand::s8:
						{
							ir::PTXS16 a = operandAsS16(threadID, instr.a);
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::s64:
						{
							ir::PTXS16 a = operandAsS16(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::u8: // fall through
					case ir::PTXOperand::b8:
						{
							ir::PTXS16 a = operandAsS16(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							ir::PTXU8 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							ir::PTXS16 a = operandAsS16(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							setRegAsU64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsS16(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsS16(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::b32: // fall through
			case ir::PTXOperand::u32:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU32 a = operandAsU32(threadID, instr.a);
							ir::PTXU8 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16:
						{
							ir::PTXU32 a = operandAsU32(threadID, instr.a);
							ir::PTXU16 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::s64: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg, 
								operandAsU32(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXU32 a = operandAsU32(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, CHAR_MAX);
							}
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXU32 a = operandAsU32(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, SHRT_MAX);
							}
							ir::PTXS16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXU32 a = operandAsU32(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, INT_MAX);
							}
							ir::PTXS32 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32: 
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsU32(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64: 
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsU32(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s32:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							ir::PTXU8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16:
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							ir::PTXU16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64: 
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							ir::PTXS16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::s64: 
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::f32: 
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsS32(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64: 
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsS32(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s64:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8: // fall through
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							ir::PTXU8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16: // fall through
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							ir::PTXU16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							ir::PTXU32 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64: 
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							ir::PTXS16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							ir::PTXS32 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s64: 
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::f32: 
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsS64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64: 
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsS64(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::b64:
			case ir::PTXOperand::u64:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							ir::PTXU8 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							ir::PTXU16 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							ir::PTXU32 d = a;
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg, 
								operandAsU64(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, CHAR_MAX);
							}
							ir::PTXS8 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, SHRT_MAX);
							}
							ir::PTXS16 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, INT_MAX);
							}
							ir::PTXS32 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s64:
						{
							ir::PTXU64 a = operandAsU64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = min(a, LLONG_MAX);
							}
							ir::PTXS64 d = a;
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32: 
						{
							setRegAsF32(threadID, instr.d.reg, 
								toF32(operandAsU64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64: 
						{
							setRegAsF64(threadID, instr.d.reg, 
								toF64(operandAsU64(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::f32:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU8 d = 0;
							if(fd > UCHAR_MAX) {
								d = UCHAR_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else {
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU16 d = 0;
							if(fd > USHRT_MAX) {
								d = USHRT_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else {
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU32 d = 0;
							if(fd > UINT_MAX) {
								d = UINT_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else {
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b64: // fall through 
					case ir::PTXOperand::u64: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU64 d = 0;
							if(fd > ULLONG_MAX) {
								d = ULLONG_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else {
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s8: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS8 d = 0;
							if(fd > CHAR_MAX) {
								d = CHAR_MAX;
							}
							else if(fd < CHAR_MIN) {
								d = CHAR_MIN;
							}
							else {
								d = fd;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS16 d = 0;
							if(fd > SHRT_MAX) {
								d = SHRT_MAX;
							}
							else if(fd < SHRT_MIN) {
								d = SHRT_MIN;
							}
							else {
								d = fd;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS32 d = 0;
							if(fd > INT_MAX) {
								d = INT_MAX;
							}
							else if(fd < INT_MIN) {
								d = INT_MIN;
							}
							else {
								d = fd;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s64: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF32 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS64 d = 0;
							if(fd > LLONG_MAX) {
								d = LLONG_MAX;
							}
							else if(fd < LLONG_MIN) {
								d = LLONG_MIN;
							}
							else {
								d = fd;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							
							a = roundToInt(a, instr.modifier, context,
								instr);
							
							setRegAsF32(threadID, instr.d.reg, 
								sat(instr.modifier, a));
						}
						break;
					case ir::PTXOperand::f64: 
						{
							ir::PTXF32 a = operandAsF32(threadID, instr.a);
							ir::PTXF64 d = toF64(a, instr.modifier);
							setRegAsF64(threadID, instr.d.reg, d);
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}					
			}
			break;
			case ir::PTXOperand::f64:
			{
				switch (instr.type) {
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8: 
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF64 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU8 d = 0;
							if(fd > UCHAR_MAX) {
								d = UCHAR_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else {
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16: 
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF64 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU16 d = 0;
							if(fd > USHRT_MAX) {
								d = USHRT_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else {
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: 
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF64 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU32 d = 0;
							if(fd > UINT_MAX) {
								d = UINT_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else
							{
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b64: // fall through 
					case ir::PTXOperand::u64: 
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0f;
							ir::PTXF64 fd = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXU64 d = 0;
							if(fd > ULLONG_MAX) {
								d = ULLONG_MAX;
							}
							else if(fd < 0) {
								d = 0;
							}
							else
							{
								d = fd;
							}
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0;
							a = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS8 d = 0;
							if(a > CHAR_MAX) {
								d = CHAR_MAX;
							}
							else if(a < CHAR_MIN) {
								d = CHAR_MIN;
							}
							else {
								d = a;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0;
							a = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS16 d = 0;
							if(a > SHRT_MAX) {
								d = SHRT_MAX;
							}
							else if(a < SHRT_MIN) {
								d = SHRT_MIN;
							}
							else {
								d = a;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0;
							a = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS32 d = 0;
							if(a > INT_MAX) {
								d = INT_MAX;
							}
							else if(a < INT_MIN) {
								d = INT_MIN;
							}
							else {
								d = a;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s64:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							if (a != a) a = 0.0;
							a = roundToInt(a, instr.modifier,
								context, instr);
							ir::PTXS64 d = 0;
							if(a > LLONG_MAX) {
								d = LLONG_MAX;
							}
							else if(a < LLONG_MIN) {
								d = LLONG_MIN;
							}
							else {
								d = a;
							}
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							a = toF32(a, instr.modifier);
							if(instr.modifier & ir::PTXInstruction::sat) {
								if (a != a) a = 0.0;
								a = min(1.0, a);
								a = max(a, 0.0);
							}
							setRegAsF32(threadID, instr.d.reg, 
								sat(instr.modifier, a));							
						}
						break;
					case ir::PTXOperand::f64: 
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsF64(threadID, instr.d.reg,
							sat(instr.modifier, a));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented", 
							context.PC, instr);
						break;
				}
			}
			break;
			default:
				throw RuntimeException("conversion not implemented", 
					context.PC, instr);
				break;
		}
		
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Cvta(CTAContext &context, 
	const ir::PTXInstruction &instr) {
		
	trace();
		
	if (!instr.toAddrSpace) {
		// convert to generic address space
		// source operand has explicitly address space
		
		switch (instr.type) {
		
		case ir::PTXOperand::u32:
		{
			ir::PTXU32 addrSpaceBase = 0;
			switch (instr.addressSpace) {
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
				}
					break;
				case ir::PTXInstruction::Global: // DO NOTHING
				case ir::PTXInstruction::Local:  // DO NOTHING
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase, 
						functionCallStack.sharedMemoryPointer());
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}
			
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				
				ir::PTXU32 srcAddr = operandAsU32(tid, instr.a) + addrSpaceBase;
				
				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU32 localMemPtr;
					if (!instr.a.isGlobalLocal) {
						hydrazine::bit_cast(localMemPtr, 
							functionCallStack.localMemoryPointer(tid));
					}
					else {
						hydrazine::bit_cast(localMemPtr, 
							functionCallStack.globalLocalMemoryPointer(tid));
					}
						
					srcAddr += localMemPtr;
				}
				
				setRegAsU32(tid, instr.d.reg, srcAddr);
			}
		}
			break;
		
		case ir::PTXOperand::u64: 
		{
			ir::PTXU64 addrSpaceBase = 0;
			switch (instr.addressSpace) {
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
				}
					break;
				case ir::PTXInstruction::Global: // DO NOTHING
				case ir::PTXInstruction::Local:  // DO NOTHING
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase, 
						functionCallStack.sharedMemoryPointer());
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}
			
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}

				ir::PTXU64 srcAddr = operandAsU64(tid, instr.a) + addrSpaceBase;
					
				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU64 localMemPtr;
					if (!instr.a.isGlobalLocal) {
						hydrazine::bit_cast(localMemPtr, 
							functionCallStack.localMemoryPointer(tid));
					}
					else {
						hydrazine::bit_cast(localMemPtr, 
							functionCallStack.globalLocalMemoryPointer(tid));
					}
						
					srcAddr += localMemPtr;
				}

				setRegAsU64(tid, instr.d.reg, srcAddr);
			}
		}
			break;
			
		default:
			throw RuntimeException("cvta instruction not valid for "
				"specified data size", context.PC, instr);
			break;
		};
	}
	else {
		// convert FROM generic address
		// dest operand has explicitly address space
		switch (instr.type) {
		case ir::PTXOperand::u32:
		{
			ir::PTXU32 addrSpaceBase = 0;
			ir::PTXU32 addrSpaceSize = 0;

			switch (instr.addressSpace) {
				case ir::PTXInstruction::Global: // DO NOTHING
				{
					addrSpaceSize = (ir::PTXU32)0xffffffff;
				}
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase, 
						functionCallStack.sharedMemoryPointer());
					addrSpaceSize = hydrazine::bit_cast<ir::PTXU32>(
						functionCallStack.sharedMemorySize());
				}
					break;
				case ir::PTXInstruction::Local: // DO NOTHING
				{

				}
					break;
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
					addrSpaceSize = kernel->constMemorySize();
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}


			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 srcAddr = operandAsU32(tid, instr.a);

				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU32 localSize = hydrazine::bit_cast<ir::PTXU32>(
						functionCallStack.localMemorySize());
					ir::PTXU32 localBase = hydrazine::bit_cast<ir::PTXU32>(
						functionCallStack.localMemoryPointer(tid));
				
					if (srcAddr >= localBase ||
						srcAddr < localSize + localBase) {
						addrSpaceSize = localSize;
						addrSpaceBase = localBase;
					}
					else {
						addrSpaceSize = hydrazine::bit_cast<ir::PTXU32>(
							functionCallStack.globalLocalMemorySize());
						addrSpaceBase = hydrazine::bit_cast<ir::PTXU32>(
							functionCallStack.globalLocalMemoryPointer(tid));
					}
				}

				if (srcAddr >= addrSpaceSize + addrSpaceBase
					|| srcAddr < addrSpaceBase) {
					throw RuntimeException("cvta instruction - source "
						"address is not part of addressed region",
						context.PC, instr);
				}

				srcAddr -= addrSpaceBase;

				setRegAsU32(tid, instr.d.reg, srcAddr);
			}
		}
			break;
		
		case ir::PTXOperand::u64: 
		{
			ir::PTXU64 addrSpaceBase = 0;
			ir::PTXU64 addrSpaceSize = 0;
			switch (instr.addressSpace) {
				case ir::PTXInstruction::Global: // DO NOTHING
				{
					addrSpaceSize = (ir::PTXU64)0xffffffffffffffffULL;
				}
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase, 
						functionCallStack.sharedMemoryPointer());
					addrSpaceSize = hydrazine::bit_cast<ir::PTXU64, size_t >(
						functionCallStack.sharedMemorySize());
				}
					break;
				case ir::PTXInstruction::Local:
				{

				}
					break;
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
					addrSpaceSize = kernel->constMemorySize();
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}
			
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 srcAddr = operandAsU64(tid, instr.a);

				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU64 localSize = hydrazine::bit_cast<ir::PTXU64>(
						functionCallStack.localMemorySize());
					ir::PTXU64 localBase = hydrazine::bit_cast<ir::PTXU64>(
						functionCallStack.localMemoryPointer(tid));
				
					if (srcAddr >= localBase ||
						srcAddr < localSize + localBase) {
						addrSpaceSize = localSize;
						addrSpaceBase = localBase;
					}
					else {
						addrSpaceSize = hydrazine::bit_cast<ir::PTXU64>(
							functionCallStack.globalLocalMemorySize());
						addrSpaceBase = hydrazine::bit_cast<ir::PTXU64>(
							functionCallStack.globalLocalMemoryPointer(tid));
					}
				}

				if (srcAddr >= addrSpaceSize + addrSpaceBase
					|| srcAddr < addrSpaceBase) {
					throw RuntimeException("cvta instruction - source "
						"address is not part of addressed region",
						context.PC, instr);
				}

				srcAddr -= addrSpaceBase;

				setRegAsU64(tid, instr.d.reg, srcAddr);
			}
		}
			break;
			
		default:
			throw RuntimeException("cvta instruction not valid for "
				"specified data size", context.PC, instr);
			break;
		};
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Div(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			if(ir::PTXInstruction::approx & instr.modifier) {
				if(issubnormal(a) || issubnormal(b))
				{
					d = a / b;
				}
				else
				{
					d = a * ( 1.0f / b );
				}
			}
			else {
				d = ftz(instr.modifier, a / b);
			}
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);
			d = a / b;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zeroa at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Ex2(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = ftz(instr.modifier, hydrazine::exp2f(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Exit(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	reconvergenceMechanism->eval_Exit(context, instr);
}


/*!

*/
void executive::CooperativeThreadArray::eval_Fma(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			ir::PTXF32 d = 0,
				a = ftz(instr.modifier, operandAsF32(tid, instr.a)), 
				b = ftz(instr.modifier, operandAsF32(tid, instr.b)), 
				c = ftz(instr.modifier, operandAsF32(tid, instr.c));

			d = ftz(instr.modifier, sat(instr.modifier, a * b + c));
			
			setRegAsF32(tid, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(tid, instr.a),
				b = operandAsF64(tid, instr.b), c = operandAsF64(tid, instr.c);
			d = a * b + c;
			setRegAsF64(tid, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}


/*!

*/
void executive::CooperativeThreadArray::eval_Isspacep(CTAContext &context,
	const ir::PTXInstruction &instr) {
	
	trace();
	switch (instr.addressSpace) {
	case ir::PTXInstruction::Local:
	{
		if (sizeof(void *) == 4) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				ir::PTXU32 localMemPtr;
				ir::PTXU32 localMemSize = functionCallStack.localMemorySize();
				hydrazine::bit_cast(localMemPtr,
					functionCallStack.localMemoryPointer(tid));
				if (ptr >= localMemPtr && localMemPtr + localMemSize > ptr) {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
			} 
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				ir::PTXU64 localMemPtr;
				ir::PTXU64 localMemSize = functionCallStack.localMemorySize();
				hydrazine::bit_cast(localMemPtr,
					functionCallStack.localMemoryPointer(tid));
				if (ptr >= localMemPtr && localMemPtr + localMemSize > ptr) {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
			} 
		}
	}
		break;
	case ir::PTXInstruction::Shared:
	{
		if (sizeof(void *) == 4) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				ir::PTXU32 sharedMemPtr;
				ir::PTXU32 sharedMemSize = functionCallStack.sharedMemorySize();
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				if (ptr >= sharedMemPtr && sharedMemPtr + sharedMemSize > ptr) {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
			} 
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 ptr = operandAsU32(tid, instr.a);
				ir::PTXU64 sharedMemPtr;
				ir::PTXU64 sharedMemSize = functionCallStack.sharedMemorySize();
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				if (ptr >= sharedMemPtr && sharedMemPtr + sharedMemSize > ptr) {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
			} 
		}
	}
		break;
	case ir::PTXInstruction::Global:
	{
		if (sizeof(void *) == 4) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				ir::PTXU32 localMemPtr;
				ir::PTXU32 localMemSize = functionCallStack.localMemorySize();
				ir::PTXU32 sharedMemPtr;
				ir::PTXU32 sharedMemSize = functionCallStack.sharedMemorySize();
				hydrazine::bit_cast(localMemPtr,
					functionCallStack.localMemoryPointer(tid));
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				if ((ptr >= sharedMemPtr && sharedMemPtr + sharedMemSize > ptr)
					|| (ptr >= localMemPtr 
					&& localMemPtr + localMemSize > ptr)) {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
			} 
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				ir::PTXU64 localMemPtr;
				ir::PTXU64 localMemSize = functionCallStack.localMemorySize();
				ir::PTXU64 sharedMemPtr;
				ir::PTXU64 sharedMemSize = functionCallStack.sharedMemorySize();
				hydrazine::bit_cast(localMemPtr,
					functionCallStack.localMemoryPointer(tid));
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				if ((ptr >= sharedMemPtr && sharedMemPtr + sharedMemSize > ptr)
					|| (ptr >= localMemPtr
						&& localMemPtr + localMemSize > ptr)) {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
			}
		}
	}
		break;
	default:
		throw RuntimeException("isspacep - invalid address space for test",
			context.PC, instr);
		break;
	}

}

/*!

*/
void executive::CooperativeThreadArray::normalLoad(int threadID, 
	const ir::PTXInstruction &instr, const char* source) {
	switch (instr.type) {
		case ir::PTXOperand::s8:
			{
				ir::PTXS8 word = *((ir::PTXS8*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsS64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::u8:
			{
				ir::PTXU8 word = *((ir::PTXU8*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::s16:		// fall through
			{
				ir::PTXS16 word = *((ir::PTXS16*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsS64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::u16:
			{
				ir::PTXU16 word = *((ir::PTXU16*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::f32:
			{
				ir::PTXF32 word = *((ir::PTXF32*)source);
#if REPORT_LD

				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsF32(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::s32:
			{
				ir::PTXS32 word = *((ir::PTXS32*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsS64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::u32:
			{
				ir::PTXU32 word = *((ir::PTXU32*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from " 
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				ir::PTXU64 word = *((ir::PTXU64*)source);
#if REPORT_LD
				report( "    Loaded " << (void*)word << " from " 
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		default:
			break;
	}		
}

void executive::CooperativeThreadArray::vectorLoad(int threadID, 
	const ir::PTXInstruction &instr, const char* source, unsigned int elementSize) {

	for (ir::PTXOperand::Array::const_iterator i = instr.d.array.begin(); 
		i != instr.d.array.end(); ++i, source += elementSize) {

		switch (instr.type) {
			case ir::PTXOperand::s8:
				{
					ir::PTXS8 word = *((ir::PTXS8*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from " 
							<< (int*)source );
#endif

					setRegAsS64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::b8:		// fall through
			case ir::PTXOperand::u8:
				{
					ir::PTXU8 word = *((ir::PTXU8*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from " 
							<< (int*)source );
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::s16:
				{
					ir::PTXS16 word = *((ir::PTXS16*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from " 
							<< (int*)source ); 
#endif

					setRegAsS64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::b16:		// fall through
			case ir::PTXOperand::u16:
				{
					ir::PTXU16 word = *((ir::PTXU16*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from " 
							<< (int*)source ); 
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::f32:
			{

				ir::PTXF32 word = *((ir::PTXF32*)((unsigned long long)source));

#if REPORT_LD
					report( "    Loaded " << word << " from " 
						<< (int*)source << " - register " << instr.d.reg
						<< "; alternative " << i->reg); 
#endif

				setRegAsF32(threadID, i->reg, word);
			}
			break;
			case ir::PTXOperand::s32:
				{
					ir::PTXS32 word = *((ir::PTXS32*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from " 
							<< (int*)source ); 
#endif

					setRegAsS64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::b32:		// fall through
			case ir::PTXOperand::u32:
				{
					ir::PTXU32 word = *((ir::PTXU32*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from " 
							<< (int*)source ); 
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::f64:		// fall through
			case ir::PTXOperand::b64:		// fall through
			case ir::PTXOperand::s64:		// fall through
			case ir::PTXOperand::u64:
				{
					ir::PTXU64 word = *((ir::PTXU64*)source);

#if REPORT_LD
						report( "    Loaded " << (void*)word << " from " 
							<< (int*)source ); 
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			default:
				break;
		}	
	}
}


/*!

*/
void executive::CooperativeThreadArray::eval_Ld(CTAContext &context, 
	const ir::PTXInstruction &instr) {

	size_t elementSize = 0;

	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				elementSize = sizeof(ir::PTXU8);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				elementSize = sizeof(ir::PTXU16);
			}
			break;
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.a.addressMode) {
				case ir::PTXOperand::Register:
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.a.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.a.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand", 
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) {
			continue;
		}

		const char *source = 0;

		switch (instr.a.addressMode) {
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
				source += getRegAsU64(threadID, instr.a.reg);
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.a.imm_uint;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand", 
					context.PC, instr);
		}

		source += instr.a.offset;
			
		switch (instr.addressSpace) {
			case ir::PTXInstruction::Param:
				{
					if (instr.a.isArgument) {
						source += kernel->scheduler->argumentMemory();
					}
					else {
						source += (ir::PTXU64) 
							functionCallStack.stackFramePointer(threadID);
					}
				}
				break;
			case ir::PTXInstruction::Const:
				{
					source += (ir::PTXU64) kernel->ConstMemory;				
				}
				break;
			case ir::PTXInstruction::Generic:
			case ir::PTXInstruction::Global:
				{	
				}
				break;
			case ir::PTXInstruction::Shared:
				{
					source += (ir::PTXU64) functionCallStack.sharedMemoryPointer();
				}
				break;
			case ir::PTXInstruction::Local:
				{
					if (instr.a.addressMode == ir::PTXOperand::Address &&
						instr.a.isGlobalLocal) {
						source += (ir::PTXU64) 
							functionCallStack.globalLocalMemoryPointer(
							threadID);
					}
					else {
						source += (ir::PTXU64) 
							functionCallStack.localMemoryPointer(threadID);
					}				
				}
				break;
			default:
				throw RuntimeException("unsupported address space", 
					context.PC, instr);
		}

		if(instr.d.vec == ir::PTXOperand::v1) {
			normalLoad(threadID, instr, source);		
		}
		else {
			vectorLoad(threadID, instr, source, elementSize);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Ldu(CTAContext &context, 
	const ir::PTXInstruction &instr) {

	size_t elementSize = 0;

	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				elementSize = sizeof(ir::PTXU8);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				elementSize = sizeof(ir::PTXU16);
			}
			break;
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.a.addressMode) {
				case ir::PTXOperand::Register:
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.a.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.a.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand", 
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();
	
	if (instr.addressSpace != ir::PTXInstruction::Global &&
		instr.addressSpace != ir::PTXInstruction::Generic) {
		throw RuntimeException("ldu - only generic and .global address "
			"space supported", context.PC, instr);
		return;
	}

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) {
			continue;
		}

		const char *source = 0;

		switch (instr.a.addressMode) {
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
				source += getRegAsU64(threadID, instr.a.reg);
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.a.imm_uint;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand", 
					context.PC, instr);
		}

		source += instr.a.offset;
		
		// these should just work if we don't add any offsets to them since 
		// it's a generic pointer

		if(instr.d.vec == ir::PTXOperand::v1) {
			normalLoad(threadID, instr, source);		
		}
		else {
			vectorLoad(threadID, instr, source, elementSize);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Lg2(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, hydrazine::log2f(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Mad24(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("instruction not implemented", context.PC, instr);
}

/*!
	Mad:

		mad[.hi,.lo,.wide][.sat].itype d, a, b, c;
		mad[.rnd][.sat].ftype d, a, b, c;

		t = a * b;
		n = bitwidth of type;
		d = t + c;            // for floating-point and .wide
		d = t<2n-1..n> + c;   // for .hi variant
		d = t<n-1..0> + c;    // for .lo variant

*/
void executive::CooperativeThreadArray::eval_Mad(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	ir::PTXOperand::DataType type = instr.type;

	switch (type) {
	case ir::PTXOperand::u16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 a = operandAsU16(threadID, instr.a); 
			ir::PTXU16 b = operandAsU16(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXU16 c = operandAsU16(threadID, instr.c);
				ir::PTXU16 t = (ir::PTXU16)(((ir::PTXU32)a * (ir::PTXU32)b) >> 16);
				ir::PTXU16 d = t + c;
				setRegAsU16(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU32 c = operandAsU32(threadID, instr.c);
				ir::PTXU32 t = (ir::PTXU32)a * (ir::PTXU32)b;
				ir::PTXU32 d = t + c;
				setRegAsU32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU16 c = operandAsU16(threadID, instr.c);
				ir::PTXU16 t = a * b;
				ir::PTXU16 d = t + c;
				setRegAsU16(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a); 
			ir::PTXU32 b = operandAsU32(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXU32 c = operandAsU32(threadID, instr.c);
				ir::PTXU32 t = (ir::PTXU32)(((ir::PTXU64)a * (ir::PTXU64)b) >> 32);
				ir::PTXU32 d = t + c;
				setRegAsU32(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU64 c = operandAsU64(threadID, instr.c);
				ir::PTXU64 t = (ir::PTXU64)a * (ir::PTXU64)b;
				ir::PTXU64 d = t + c;
				setRegAsU64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU32 c = operandAsU32(threadID, instr.c);
				ir::PTXU32 t = a * b;
				ir::PTXU32 d = t + c;
				setRegAsU32(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::u64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU64 d = 0,
				a = operandAsU64(threadID, instr.a), 
				b = operandAsU64(threadID, instr.b), 
				c = operandAsU64(threadID, instr.c);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXU64 hi = 0;
				ir::PTXU64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi + c;
			}
			else {
				ir::PTXU64 t = a * b;
				d = t + c;
			}
			setRegAsU64(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 a = operandAsS16(threadID, instr.a); 
			ir::PTXS16 b = operandAsS16(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXS16 c = operandAsS16(threadID, instr.c);
				ir::PTXS16 t = (ir::PTXS16)(((ir::PTXS32)a * (ir::PTXS32)b) >> 16);
				ir::PTXS16 d = t + c;
				setRegAsS16(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS32 c = operandAsS32(threadID, instr.c);
				ir::PTXS32 t = (ir::PTXS32)a * (ir::PTXS32)b;
				ir::PTXS32 d = t + c;
				setRegAsS32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS16 c = operandAsS16(threadID, instr.c);
				ir::PTXS16 t = a * b;
				ir::PTXS16 d = t + c;
				setRegAsS16(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 a = operandAsS32(threadID, instr.a); 
			ir::PTXS32 b = operandAsS32(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXS32 c = operandAsS32(threadID, instr.c);
				ir::PTXS32 t = (ir::PTXS32)(((ir::PTXS64)a * (ir::PTXS64)b) >> 32);
				ir::PTXS32 d = 0;
				if (instr.modifier & ir::PTXInstruction::sat) {
					ir::PTXS64 td = (ir::PTXS64)t + (ir::PTXS64)c;
					td = max(td, (ir::PTXS64)INT_MIN);
					td = min(td, (ir::PTXS64)INT_MAX);
					d = td;
				}
				else {
					d = t + c;
				}
				setRegAsS32(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS64 c = operandAsS64(threadID, instr.c);
				ir::PTXS64 t = (ir::PTXS64)a * (ir::PTXS64)b;
				ir::PTXS64 d = t + c;
				setRegAsS64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS32 c = operandAsS32(threadID, instr.c);
				ir::PTXS32 t = a * b;
				ir::PTXS32 d = t + c;
				setRegAsS32(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::s64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d = 0,
				a = operandAsS64(threadID, instr.a), 
				b = operandAsS64(threadID, instr.b), 
				c = operandAsS64(threadID, instr.c);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXS64 hi = 0;
				ir::PTXS64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi + c;
			}
			else {
				ir::PTXS64 t = a * b;
				d = t + c;
			}
			setRegAsS64(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::f32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d = 0,
				a = ftz(instr.modifier, operandAsF32(threadID, instr.a)), 
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b)), 
				c = ftz(instr.modifier, operandAsF32(threadID, instr.c));

			d = ftz(instr.modifier, sat(instr.modifier, a * b + c));
			
			setRegAsF32(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::f64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF64 d = 0,
				a = operandAsF64(threadID, instr.a), 
				b = operandAsF64(threadID, instr.b), 
				c = operandAsF64(threadID, instr.c);

			d = a * b + c;
			if (instr.modifier & ir::PTXInstruction::sat) {
				if (d < 0) d = 0; else if (d > 1) d = 1;
			}
			setRegAsF64(threadID, instr.d.reg, d);
		}
	} break;

	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Max(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a),
				b = operandAsF32(threadID, instr.b);

			if(hydrazine::isnan(a))
			{
				d = ftz(instr.modifier, b);
			}
			else if(hydrazine::isnan(b))
			{
				d = ftz(instr.modifier, a);
			}
			else
			{
				d = ftz(instr.modifier, a > b ? a : b);
			}

			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);

			if(hydrazine::isnan(a))
			{
				d = b;
			}
			else if(hydrazine::isnan(b))
			{
				d = a;
			}
			else
			{
				d = (a > b ? a : b);
			}

			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Min(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a),
				b = operandAsF32(threadID, instr.b);

			if(hydrazine::isnan(a))
			{
				d = ftz(instr.modifier, b);
			}
			else if(hydrazine::isnan(b))
			{
				d = ftz(instr.modifier, a);
			}
			else
			{
				d = ftz(instr.modifier, (a < b ? a : b));
			}

			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);

			if(hydrazine::isnan(a))
			{
				d = b;
			}
			else if(hydrazine::isnan(b))
			{
				d = a;
			}
			else
			{
				d = (a < b ? a : b);
			}

			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Membar(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	/*! No need to do anything here. */
}

////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Mov(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();

	if (instr.a.addressMode == ir::PTXOperand::Register) {
		eval_Mov_reg(context, instr);
	}
	else if (instr.a.addressMode == ir::PTXOperand::Special) {
		eval_Mov_sreg(context, instr);		
	}
	else if (instr.a.addressMode == ir::PTXOperand::Immediate) {
		eval_Mov_imm(context, instr);		
	}
	else if (instr.a.addressMode == ir::PTXOperand::Address) {
		eval_Mov_addr(context, instr);
	}
	else if (instr.a.addressMode == ir::PTXOperand::FunctionName) {
		eval_Mov_func(context, instr);
	}
	else {
		throw RuntimeException(std::string("unimplemented address mode ") 
			+ ir::PTXOperand::toString(instr.a.addressMode), context.PC, instr);
	}
}

template<typename D, typename S>
static D gather(S s0, S s1)
{
	D result = s1;
	result <<= sizeof(S) * 8;
	result |= s0;
	return result;
}

template<typename D, typename S>
static D gather(S s0, S s1, S s2, S s3)
{
	D result = s3;
	result <<= sizeof(S) * 8;
	result |= s2;
	result <<= sizeof(S) * 8;
	result |= s1;
	result <<= sizeof(S) * 8;
	result |= s0;
	return result;
}

template<typename D, unsigned int index, unsigned int bytes, typename S>
static D scatter(S s)
{
	ir::PTXU64 mask = 1;
	mask <<= bytes * 8;
	mask -= 1;
	s >>= index * bytes * 8;
	s &= mask;
	return s;
}

void executive::CooperativeThreadArray::eval_Mov_reg(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		
		switch (instr.type) {
		case ir::PTXOperand::u16:
		case ir::PTXOperand::s16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				setRegAsU16(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::b16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				if (instr.a.array.size() == 0) {
					a = operandAsU32(threadID, instr.a);
				}
				else {
					assert(instr.a.array.size() == 2);
					a = gather<ir::PTXB16>(operandAsB8(threadID, instr.a.array[0]), 
						operandAsB8(threadID, instr.a.array[1]));
				}
				if (instr.d.array.size() == 0) {
					setRegAsB16(threadID, instr.d.reg, a);
				}
				else {
					assert(instr.d.array.size() == 2);
					setRegAsB8(threadID, instr.d.array[0].reg, 
						scatter<ir::PTXB8, 0, 1>(a));
					setRegAsB8(threadID, instr.d.array[1].reg, 
						scatter<ir::PTXB8, 1, 1>(a));
				}
			}
			break;
		case ir::PTXOperand::u32:
		case ir::PTXOperand::s32:
			{
				ir::PTXU32 a = operandAsU32(threadID, instr.a);
				setRegAsU32(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::b32:
			{
				ir::PTXU32 a = operandAsU32(threadID, instr.a);
				if (instr.a.array.size() == 0) {
					a = operandAsU32(threadID, instr.a);
				}
				else if (instr.a.array.size() == 2) {
					a = gather<ir::PTXB32>(operandAsB16(threadID, 
						instr.a.array[0]), operandAsB16(threadID, 
						instr.a.array[1]));
				}
				else {
					assert(instr.a.array.size() == 4);
					a = gather<ir::PTXB32>(operandAsB8(threadID, instr.a.array[0]), 
						operandAsB8(threadID, instr.a.array[1]), 
						operandAsB8(threadID, instr.a.array[2]), 
						operandAsB8(threadID, instr.a.array[3]));
				}
				if (instr.d.array.size() == 0) {
					setRegAsU32(threadID, instr.d.reg, a);
				}
				else if (instr.d.array.size() == 2) {
					setRegAsU16(threadID, instr.d.array[0].reg, 
						scatter<ir::PTXB16, 0, 2>(a));
					setRegAsU16(threadID, instr.d.array[1].reg, 
						scatter<ir::PTXB16, 1, 2>(a));
				}
				else {
					assert(instr.d.array.size() == 4);
					setRegAsB8(threadID, instr.d.array[0].reg, 
						scatter<ir::PTXB8, 0, 1>(a));
					setRegAsB8(threadID, instr.d.array[1].reg, 
						scatter<ir::PTXB8, 1, 1>(a));
					setRegAsB8(threadID, instr.d.array[2].reg, 
						scatter<ir::PTXB8, 2, 1>(a));
					setRegAsB8(threadID, instr.d.array[3].reg, 
						scatter<ir::PTXB8, 3, 1>(a));
				}
			}
			break;
		case ir::PTXOperand::u64:
		case ir::PTXOperand::s64:
			{
				ir::PTXU64 a = operandAsU64(threadID, instr.a);
				setRegAsU64(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::b64:
			{
				ir::PTXU64 a;
				if (instr.a.array.size() == 0) {
					a = operandAsU64(threadID, instr.a);
				}
				else if (instr.a.array.size() == 2) {
					a = gather<ir::PTXB64>(operandAsB32(threadID, 
						instr.a.array[0]), 
						operandAsB32(threadID, instr.a.array[1]));
				}
				else {
					assert(instr.a.array.size() == 4);
					a = gather<ir::PTXB64>(
						operandAsB16(threadID, instr.a.array[0]), 
						operandAsB16(threadID, instr.a.array[1]), 
						operandAsB16(threadID, instr.a.array[2]), 
						operandAsB16(threadID, instr.a.array[3]));
				}
				if (instr.d.array.size() == 0) {
					setRegAsB64(threadID, instr.d.reg, a);
				}
				else if (instr.d.array.size() == 2) {
					setRegAsB32(threadID, instr.d.array[0].reg, 
						scatter<ir::PTXB32, 0, 4>(a));
					setRegAsB32(threadID, instr.d.array[1].reg, 
						scatter<ir::PTXB32, 1, 4>(a));
				}
				else {
					assert(instr.d.array.size() == 4);
					setRegAsB16(threadID, instr.d.array[0].reg, 
						scatter<ir::PTXB16, 0, 2>(a));
					setRegAsB16(threadID, instr.d.array[1].reg, 
						scatter<ir::PTXB16, 1, 2>(a));
					setRegAsB16(threadID, instr.d.array[2].reg, 
						scatter<ir::PTXB16, 2, 2>(a));
					setRegAsB16(threadID, instr.d.array[3].reg, 
						scatter<ir::PTXB16, 3, 2>(a));
				}
			}
			break;
		case ir::PTXOperand::f32:
			{
				setRegAsF32(threadID, instr.d.reg,
					operandAsF32(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::f64:
			{
				setRegAsF64(threadID, instr.d.reg,
					operandAsF64(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::pred:
			{
				setRegAsPredicate(threadID, instr.d.reg,
					operandAsU64(threadID, instr.a));
			}
			break;
		default:
			break;
		}
	}
}

void executive::CooperativeThreadArray::eval_Mov_sreg(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	
	if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 d = operandAsU16(threadID, instr.a);
			setRegAsU16(threadID, instr.d.reg, d);			
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d = operandAsU32(threadID, instr.a);
			setRegAsU32(threadID, instr.d.reg, d);			
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d = operandAsS32(threadID, instr.a);
			setRegAsS32(threadID, instr.d.reg, d);			
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Mov_imm(CTAContext &context,
	const ir::PTXInstruction &instr) {
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		
		switch (instr.d.type) {
		case ir::PTXOperand::u16:
		case ir::PTXOperand::s16:
		case ir::PTXOperand::b16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				setRegAsU16(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::u32:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::b32:
			{
				ir::PTXU32 a = operandAsU32(threadID, instr.a);
				setRegAsU32(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::u64:
		case ir::PTXOperand::s64:
		case ir::PTXOperand::b64:
			{
				ir::PTXU64 a = operandAsU64(threadID, instr.a);
				setRegAsU64(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::f32:
			{
				setRegAsF32(threadID, instr.d.reg,
					operandAsF32(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::f64:
			{
				setRegAsF64(threadID, instr.d.reg,
					operandAsF64(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::pred:
			{
				setRegAsPredicate(threadID, instr.d.reg,
					operandAsU64(threadID, instr.a));
			}
			break;
		default:
			break;
		}
	}
}

void executive::CooperativeThreadArray::eval_Mov_indirect(CTAContext &context,
	const ir::PTXInstruction &instr) {
	
}

void executive::CooperativeThreadArray::eval_Mov_addr(CTAContext &context,
	const ir::PTXInstruction &instr) {
	eval_Mov_imm(context, instr);
}

void executive::CooperativeThreadArray::eval_Mov_func(CTAContext &context,
	const ir::PTXInstruction &instr) {
	if (instr.branchTargetInstruction == -1) {
		kernel->lazyLink(context.PC, instr.a.identifier);
	}
	
	const ir::PTXInstruction& jittedInstr = currentInstruction(context);
	
	currentEvent.instruction = &jittedInstr;

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, jittedInstr)) continue;
		setRegAsU64(threadID, jittedInstr.d.reg,
			jittedInstr.branchTargetInstruction);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Mul24(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();

	if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 a = operandAsU32(threadID, instr.a), b = operandAsU32(threadID, instr.b);
			ir::PTXU64 dw = ( a & 0x00ffffff ) * ( b & 0x00ffffff );
			ir::PTXU32 d;
			if (instr.modifier & ir::PTXInstruction::hi) {
				d = ((dw >> 12) & 0xffffffff);
			}
			else {
				d = (dw & 0xffffffff);
			}
			setRegAsU32(threadID, instr.d.reg, d);
		
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 a = operandAsS32(threadID, instr.a), b = operandAsS32(threadID, instr.b);
			
			if( a < 0 )
			{
				a |= 0xffffffffff000000ULL;
			}
			else
			{
				a &= 0x00ffffff;
			}
			if( b < 0 )
			{
				b |= 0xffffffffff000000ULL;
			}
			else
			{
				b &= 0x00ffffff;
			}
			
			ir::PTXS64 dw = a * b;
			ir::PTXS32 d;
			if (instr.modifier & ir::PTXInstruction::hi) {
				d = ((dw >> 12) & 0xffffffff);
			}
			else {
				d = dw;
			}
			setRegAsS32(threadID, instr.d.reg, d);

		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Mul(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			d = ftz(instr.modifier, sat(instr.modifier, a * b));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a), b = operandAsF64(threadID, instr.b);
			d = a * b;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 a = operandAsU16(threadID, instr.a), b = operandAsU16(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU32 d = a * b;
				setRegAsU32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU32 dw = a * b;
				ir::PTXU16 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 16) & 0x0ffff);
				}
				else {
					d = (dw & 0x0ffff);
				}
				setRegAsU16(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 a = operandAsU32(threadID, instr.a), b = operandAsU32(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU64 d = a * b;
				setRegAsU64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU64 dw = a * b;
				ir::PTXU32 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 32) & 0x0ffffffff);
				}
				else {
					d = (dw & 0x0ffffffff);
				}
				setRegAsU32(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 a = operandAsS16(threadID, instr.a), b = operandAsS16(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS32 d = a * b;
				setRegAsS32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS32 dw = a * b;
				ir::PTXS16 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 16) & 0x0ffff);
				}
				else {
					d = (dw & 0x0ffff);
				}
				setRegAsS16(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 a = operandAsS32(threadID, instr.a), b = operandAsS32(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS64 d = a * b;
				setRegAsS64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS64 dw = a * b;
				ir::PTXS32 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 32) & 0x0ffffffff);
				}
				else {
					d = dw;
				}
				setRegAsS32(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		if (instr.modifier & ir::PTXInstruction::lo) {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXS64 d, a = operandAsS64(threadID, instr.a), 
					b = operandAsS64(threadID, instr.b);
				d = a * b;
				setRegAsS64(threadID, instr.d.reg, d);
			}
		}
		else {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXS64 d, a = operandAsS64(threadID, instr.a), 
					b = operandAsS64(threadID, instr.b);
				ir::PTXS64 hi = 0;
				ir::PTXS64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi;
				setRegAsS64(threadID, instr.d.reg, d);
			}
		}
	} 
	else if (instr.type == ir::PTXOperand::u64) {
		if (instr.modifier & ir::PTXInstruction::lo) {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 d, a = operandAsU64(threadID, instr.a), 
					b = operandAsU64(threadID, instr.b);
				d = a * b;
				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		else {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 d, a = operandAsU64(threadID, instr.a), 
					b = operandAsU64(threadID, instr.b);
				ir::PTXU64 hi = 0;
				ir::PTXU64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi;
				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Neg(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = ftz(instr.modifier, -a);
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = -a;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a);
			d = -a;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a);
			d = -a;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a);
			d = -a;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Not(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			bool d, a = operandAsPredicate(threadID, instr.a);
			d = !a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a);
			d = ~a;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = ~a;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB64 d, a = operandAsB64(threadID, instr.a);
			d = ~a;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Or(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			bool d, a = operandAsPredicate(threadID, instr.a), 
				b = operandAsPredicate(threadID, instr.b);
			d = b || a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a), 
				b = operandAsB16(threadID, instr.b);
			d = a | b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a), 
				b = operandAsB32(threadID, instr.b);
			d = a | b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB64 d, a = operandAsB64(threadID, instr.a), 
				b = operandAsB64(threadID, instr.b);
			d = a | b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Pmevent(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	/*! No need to do anything here. */
}

/*!

*/
void executive::CooperativeThreadArray::eval_Popc(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = hydrazine::popc(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXB32 d = hydrazine::popc(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Prefetch(CTAContext &context, const ir::PTXInstruction &instr) {
	report(instr.toString());
	
	if (traceEvents) {
		currentEvent.memory_size = 4;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.d.addressMode) {
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.d.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.d.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand", 
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}
	
	trace();
}

void executive::CooperativeThreadArray::eval_Prefetchu(CTAContext &context, const ir::PTXInstruction &instr) {
	
	if (traceEvents) {
		currentEvent.memory_size = 4;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.d.addressMode) {
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.d.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.d.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand", 
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}
	
	trace();
}

/*!

*/
void executive::CooperativeThreadArray::eval_Prmt(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b32) {
		switch( instr.permuteMode ) {
			case ir::PTXInstruction::DefaultPermute: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<hydrazine::DefaultPermute>(
						a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::ForwardFourExtract: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<
						hydrazine::ForwardFourExtract>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::BackwardFourExtract: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<
						hydrazine::BackwardFourExtract>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::ReplicateEight: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = 0;
					d = hydrazine::permute<hydrazine::ReplicateEight>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::EdgeClampLeft: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = 0;
					d = hydrazine::permute<hydrazine::EdgeClampLeft>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::EdgeClampRight: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = 0;
					d = hydrazine::permute<hydrazine::EdgeClampRight>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::ReplicateSixteen: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;
			
					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<hydrazine::ReplicateSixteen>(
						a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
		}
	}	
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Rcp(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, 1.0f/a);
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = 1.0/a;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Red(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("instruction not implemented", context.PC, instr);
}

/*!

*/
void executive::CooperativeThreadArray::eval_Rem(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Modulus by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a % b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Modulus by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a % b;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Modulus by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a % b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Modulus by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a % b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Modulus by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a % b;
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a), 
				b = operandAsU64(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Modulus by zero at: " 
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a % b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Ret(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	
	if (functionCallStack.isTheCurrentFrameMain()) {
		eval_Exit(context, instr);
		return;
	}
	
	int returnedPC = functionCallStack.returnPC();
	
	reportE(REPORT_RET, "Returned from function call at PC " 
		<< returnedPC );
	const ir::PTXInstruction& call = kernel->instructions[returnedPC];
	reportE(REPORT_RET, " Previous stack size (" 
		<< functionCallStack.callerFrameSize() );
	unsigned int offset = 0;
	for (ir::PTXOperand::Array::const_iterator argument = call.d.array.begin();
		argument != call.d.array.end(); ++argument) {
		for (int threadID = 0; threadID != threadCount; ++threadID) {
			if (!context.predicated(threadID, instr)) continue;
			
			char* callerPointer =
				(char*)functionCallStack.callerFramePointer(threadID);
			char* pointer =
				(char*)functionCallStack.stackFramePointer(threadID);
			
			reportE(REPORT_RET, " For thread " << threadID << " copying " 
				<< argument->toString() << " ["
				<< ir::PTXOperand::bytes(argument->type)
				<< " bytes] from new frame at " 
				<< offset << " to caller frame at " 
				<< argument->offset
				<< " - destination: "
				<< (void *)(callerPointer + argument->offset)
				<< " - caller stack size: "
					<< functionCallStack.callerFrameSize());
			
			std::memcpy(callerPointer + argument->offset, pointer + offset, 
				ir::PTXOperand::bytes(argument->type));
			
		}
		offset += ir::PTXOperand::bytes(argument->type);
	}

	functionCallStack.popFrame();
	reconvergenceMechanism->pop();
	
	// if we returned all the way back, copy operands into registers
	CTAContext &returnedContext = getActiveContext();
	
	if (returnedContext.PC == returnedPC + 1) {
		for (ir::PTXOperand::Array::const_iterator
			argument = call.d.array.begin();
			argument != call.d.array.end(); ++argument) {
			if (argument->addressMode != ir::PTXOperand::Register) continue;
			
			for (int threadID = 0; threadID != threadCount; ++threadID) {
				if (!returnedContext.predicated(threadID, instr)) continue;
			
				char* frame =
					(char*)functionCallStack.stackFramePointer(threadID);
			
				ir::PTXU64 data = 0;

				std::memcpy(&data, frame + argument->offset, 
					ir::PTXOperand::bytes(argument->type));
				
				setRegAsU64(threadID, argument->reg, data);
			}
		}
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Rsqrt(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, 1.0f/(ir::PTXF32)std::sqrt(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = 1.0/sqrt(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	sad.type d, a, b, c;

	d = c + ((a<b) ? b-a : a-b);

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

*/		
void executive::CooperativeThreadArray::eval_Sad(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::u16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b),
				c = operandAsU16(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsU16(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b),
				c = operandAsU32(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsU32(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::u64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b),
				c = operandAsU64(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsU64(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b),
				c = operandAsS16(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsS16(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b),
				c = operandAsS32(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsS32(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b),
				c = operandAsS64(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsS64(threadID, instr.d.reg, d);
		}
	} break;
	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!
	Select between source operands according to the predicate value
selp.type d, a, b, c;

.type = { .b16, .b32, .b64,
          .u16, .u32, .u64,
          .s16, .s32, .s64,
                .f32, .f64 };

*/
void executive::CooperativeThreadArray::eval_SelP(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	switch (instr.type) {
		case ir::PTXOperand::b16:	// fall through
		case ir::PTXOperand::s16:	// fall through
		case ir::PTXOperand::u16:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXU16 a = operandAsU16(threadID, instr.a),
					b = operandAsU16(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);
				
				ir::PTXU16 d = (c ? a : b);
				
				setRegAsU16(threadID, instr.d.reg, d);
			}
		}
			break;
		
		case ir::PTXOperand::s32:	// fall through
		case ir::PTXOperand::b32:	// fall through
		case ir::PTXOperand::u32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXU32 a = operandAsU32(threadID, instr.a),
					b = operandAsU32(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);
				
				ir::PTXU32 d = (c ? a : b);
				
				setRegAsU32(threadID, instr.d.reg, d);
			}
		}
		break;
		
		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXF32 a = operandAsF32(threadID, instr.a),
					b = operandAsF32(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);
				
				ir::PTXF32 d = (c ? a : b);
				
				setRegAsF32(threadID, instr.d.reg, d);
			}
		}
		break;
		
		case ir::PTXOperand::f64:	// fall through
		case ir::PTXOperand::s64:	// fall through
		case ir::PTXOperand::b64:	// fall through
		case ir::PTXOperand::u64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);
				ir::PTXU64 d = (c ? a : b);

				reportE( REPORT_SELP, "  thread[" << threadID << "] d = 0x" 
					<< std::hex << d << ", " << instr.d.identifier << " = " 
					<< getRegAsU64(threadID, instr.d.reg) << std::dec);
				
				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	Compare two numeric values with a relational operator 
	and [optionally] combine this result
	with a predicate value by applying a boolean operator

setp.CmpOp.type p[|q], a, b;
setp.CmpOp.BoolOp.type p[|q], a, b, [!]c;

.type = { .b16, .b32, .b64,
	.u16, .u32, .u64,
	.s16, .s32, .s64,
	.f32, .f64 };

*/
void executive::CooperativeThreadArray::eval_SetP(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	
	switch (instr.type) {
		
		// unsigned int types [extended to 64-bit uint]
		case ir::PTXOperand::pred:
		case ir::PTXOperand::b16:
		case ir::PTXOperand::b32:
		case ir::PTXOperand::b64:
		case ir::PTXOperand::u16:
		case ir::PTXOperand::u32:
		case ir::PTXOperand::u64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				bool c = true;
				bool t = false;
				
				ir::PTXU64 a, b;

				switch (instr.type) {
					case ir::PTXOperand::pred:
						a = (ir::PTXU64)operandAsPredicate(threadID, instr.a) & 0x1;
						b = (ir::PTXU64)operandAsPredicate(threadID, instr.b) & 0x1;
						break;
					case ir::PTXOperand::s16:
					case ir::PTXOperand::b16:
					case ir::PTXOperand::u16:
						a = (ir::PTXU64)operandAsU16(threadID, instr.a) & 0x0ffff;
						b = (ir::PTXU64)operandAsU16(threadID, instr.b) & 0x0ffff;
						break;
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::b32:
						a = (ir::PTXU64)operandAsU32(
							threadID, instr.a) & 0x0ffffffff;
						b = (ir::PTXU64)operandAsU32(
							threadID, instr.b) & 0x0ffffffff;
						break;
					default:
						a = operandAsU64(threadID, instr.a);
						b = operandAsU64(threadID, instr.b);
						break;
				}

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;
					default:
						throw RuntimeException(
							"invalid comparison operator for unsigned int type",
							context.PC, instr);
				}
				
				// now apply the bool op
				bool p = false, q = false;
				
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t  || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t  && !c) || (!t && c);
						q = (!t && !c) || (t  && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a 
					<< ", " << instr.b.identifier << " = " << b 
					<< " condition = " << t << ", input = " << c << " " 
					<< instr.d.identifier << " = " << p << ", q = " << q );
									
				setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;
		
		// signed int types [extended to 64-bit int]
		case ir::PTXOperand::s16:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::s64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				bool c = true;	// read operator somehow
				bool t = false;

				ir::PTXS64 a, b;

				switch (instr.type) {
					case ir::PTXOperand::s16:
						a = (ir::PTXS64)operandAsS16(threadID, instr.a);
						b = (ir::PTXS64)operandAsS16(threadID, instr.b);
						break;
					case ir::PTXOperand::s32:
						a = (ir::PTXS64)operandAsS32(threadID, instr.a);
						b = (ir::PTXS64)operandAsS32(threadID, instr.b);
						break;
					default:
						a = operandAsS64(threadID, instr.a);
						b = operandAsS64(threadID, instr.b);
						break;
				}			
				
				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will
				// get this wrong twice or less
				switch (instr.comparisonOperator) {
				
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;
					
					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}
				
				// now apply the bool op
				bool p = false, q = false;
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t && !c) || (!t && c);
						q = (!t && !c) || (t && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a 
					<< ", " << instr.b.identifier << " = " << b 
					<< " condition = " << t << ", input = " << c << " " 
					<< instr.d.identifier << " = " << p << ", q = " << q );
				
				setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;
		
		// single-precision float
		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXF32 a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
					b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
				bool c = true;	// read operator somehow
				bool t = false;
				
				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;
					case ir::PTXInstruction::Nan:
						t = hydrazine::isnan(a) || hydrazine::isnan(b);					
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}
								
				// now apply the bool op
				bool p = false, q = false;
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t && !c) || (!t && c);
						q = (!t && !c) || (t && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a 
					<< ", " << instr.b.identifier << " = " << b 
					<< " condition = " << t << ", input = " << c << " " 
					<< instr.d.identifier << " = " << p << ", q = " << q );
				
				setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;
		
		// double-precision float
		case ir::PTXOperand::f64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXF64 a = operandAsF64(threadID, instr.a),
					b = operandAsF64(threadID, instr.b);
				bool c = true;
				bool t = false;
				
				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
				
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;
					case ir::PTXInstruction::Nan:
						t = hydrazine::isnan(a) || hydrazine::isnan(b);					
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}				

				
				// now apply the bool op
				bool p = false, q = false;
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t && !c) || (!t && c);
						q = (!t && !c) || (t && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a 
					<< ", " << instr.b.identifier << " = " << b 
					<< " condition = " << t << ", input = " << c << " " 
					<< instr.d.identifier << " = " << p << ", q = " << q );
				
				setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;
		
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	ir::PTX set instruction

	t = (a CmpOp b) ? 1 : 0;
	if (isFloat(dtype))
		d = BoolOp(t, c) ? 1.0f : 0x00000000;
	else
		d = BoolOp(t, c) ? 0xFFFFFFFF : 0x00000000;

*/
void executive::CooperativeThreadArray::eval_Set(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	
	switch (instr.a.type) {
		
		// unsigned int types [extended to 64-bit uint]
		case ir::PTXOperand::b16:
		case ir::PTXOperand::b32:
		case ir::PTXOperand::b64:
		case ir::PTXOperand::u16:
		case ir::PTXOperand::u32:
		case ir::PTXOperand::u64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				bool c = true;	// read predicate somehow
				bool t = false;
				
				ir::PTXU64 a, b;

				switch (instr.a.type) {
					case ir::PTXOperand::u16:
					case ir::PTXOperand::b16:
					case ir::PTXOperand::s16:
						a = (ir::PTXU64)operandAsU16(threadID, instr.a);
						b = (ir::PTXU64)operandAsU16(threadID, instr.b);
						break;
					case ir::PTXOperand::b32:
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						a = (ir::PTXU64)operandAsU32(threadID, instr.a);
						b = (ir::PTXU64)operandAsU32(threadID, instr.b);
						break;
					default:
						a = operandAsU64(threadID, instr.a);
						b = operandAsU64(threadID, instr.b);
						break;
				}	

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;
					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}
				
				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;
		
		// signed int types [extended to 64-bit int]
		case ir::PTXOperand::s16:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::s64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				bool c = true;	// read operator somehow
				bool t = false;

				ir::PTXS64 a, b;

				switch (instr.a.type) {
					case ir::PTXOperand::s16:
					case ir::PTXOperand::u16:
					case ir::PTXOperand::b16:
						a = (ir::PTXS64)operandAsS16(threadID, instr.a);
						b = (ir::PTXS64)operandAsS16(threadID, instr.b);
						break;
					case ir::PTXOperand::b32:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::s32:
						a = (ir::PTXS64)operandAsS32(threadID, instr.a);
						b = (ir::PTXS64)operandAsS32(threadID, instr.b);
						break;
					default:
						a = operandAsS64(threadID, instr.a);
						b = operandAsS64(threadID, instr.b);
						break;
				}			
				
				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
				
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;
					
					default:
						throw RuntimeException("invalid comparison operator "
						"for unsigned int type", context.PC, instr);
				}
				
				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;
		
		// single-precision float
		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXF32 a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
					b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
				bool c = true;	// read operator somehow
				bool t = false;
				
				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ne:
						t = !hydrazine::isnan(b) && !hydrazine::isnan(a) && (a != b);
						break;
					
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;
					
					case ir::PTXInstruction::Nan:
						break;
						
					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}
				
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Nan:
						// if either is NaN, set t to true
						t = (hydrazine::isnan(a) || hydrazine::isnan(b) || t);
						break;
					default:
						break;
				}
				
				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;
		
		// double-precision float
		case ir::PTXOperand::f64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				
				ir::PTXF64 a = operandAsF64(threadID, instr.a),
					b = operandAsF64(threadID, instr.b);
				bool c = true;
				bool t = false;
				
				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}
				
				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
				
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;
					
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;
					case ir::PTXInstruction::Nan:
						t = hydrazine::isnan(a) || hydrazine::isnan(b);
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}				
				
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Num:
					case ir::PTXInstruction::Nan:
						// if either is NaN, set t to true
						t = (hydrazine::isnan(a) || hydrazine::isnan(b) || t);
						break;
					default:
						break;
				}
				
				
				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;
		
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Shl(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	unsigned int b = 0;
	if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB16 d, a;
			
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}		
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 16 )
			{
				b = 16;
			}
			d = a << b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 32 )
			{
				b = 32;
			}
			d = a << b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 64 )
			{
				b = 64;
			}
			d = a << b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Shr(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	unsigned int b = 0;
	if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB16 d, a;
			
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 16 )
			{
				b = 16;
			}
			d = a >> b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 32 )
			{
				b = 32;
			}
			d = a >> b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 64 )
			{
				b = 64;
			}
			d = a >> b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_int;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsS16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 16 )
			{
				b = 16;
			}
			d = a >> b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_int;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsS32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 32 )
			{
				b = 32;
			}
			d = a >> b;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_int;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsS64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 64 )
			{
				b = 64;
			}
			d = a >> b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU16 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsU16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 16 )
			{
				b = 16;
			}
			d = a >> b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsU32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 32 )
			{
				b = 32;
			}
			d = a >> b;
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsU64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type", 
					context.PC, instr);
			}
			if( b > 64 )
			{
				b = 64;
			}
			d = a >> b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Sin(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = ftz(instr.modifier, (ir::PTXF32)sin(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = sin(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	slct.dtype.ctype d, a, b, c;

	.dtype = { .b16, .b32, .b64,
		         .u16, .u32, .u64,
		         .s16, .s32, .s64,
		               .f32, .f64 };

	.ctype = { .s32, .f32 };

	d = (c >= 0) ? a : b;
*/
void executive::CooperativeThreadArray::eval_SlCt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	assert(instr.opcode == ir::PTXInstruction::SlCt);

	switch (instr.c.type) {
		case ir::PTXOperand::u32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXS32 c = operandAsU32(threadID, instr.c);

				ir::PTXU64 d = ((c >= 0) ? a : b);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		break;
		case ir::PTXOperand::s32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXS32 c = operandAsS32(threadID, instr.c);

				ir::PTXU64 d = ((c >= 0) ? a : b);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		break;
		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXF32 c = ftz(instr.modifier, operandAsF32(threadID, instr.c));

				ir::PTXU64 d = ((c >= 0) ? a : b);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		break;
		default:
			throw RuntimeException("source data type not permitted",
				context.PC, instr);
			break;
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Sqrt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	assert(instr.opcode == ir::PTXInstruction::Sqrt);

	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			
			if(a < 0.0f || hydrazine::isnan(a))
			{
				d = std::numeric_limits<float>::signaling_NaN();
			}
			else
			{
				d = std::sqrt(a);
			}
			
			setRegAsF32(threadID, instr.d.reg, ftz(instr.modifier, d));
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = sqrt(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

////////////////////////////////////////////////////////////////////////////////

void executive::CooperativeThreadArray::normalStore(int threadID, 
	const ir::PTXInstruction &instr, char* source) {
	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				ir::PTXU8 word = getRegAsU8(threadID, instr.a.reg);
#if REPORT_ST
				report( "    Storing " << word << " to " 
					<< (int*)source );
#endif
				*((ir::PTXU8*)source) = word;
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				ir::PTXU16 word = getRegAsU16(threadID, instr.a.reg);
#if REPORT_ST
				report( "    Storing " << word << " to " 
					<< (int*)source );
#endif
				*((ir::PTXU16*)source) = word;
			}
			break;
		case ir::PTXOperand::f32:
			{
				ir::PTXF32 word = getRegAsF32(threadID, instr.a.reg);
#if REPORT_ST
				report( "    Storing " << word << " to " 
					<< (int*)source );
#endif
				*((ir::PTXF32*)source) = word;
			}
			break;
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				ir::PTXU32 word = getRegAsU32(threadID, instr.a.reg);
#if REPORT_ST
				report( "    Storing " << word << " to " 
					<< (int*)source );
#endif
				*((ir::PTXU32*)source) = word;
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				ir::PTXU64 word = getRegAsU64(threadID, instr.a.reg);
#if REPORT_ST
				report( "    Storing " << (void*)word << " to " 
					<< (int*)source );
#endif
				*((ir::PTXU64*)source) = word;
			}
			break;
		default:
			break;
	}
		
}

void executive::CooperativeThreadArray::vectorStore(int threadID, 
	const ir::PTXInstruction &instr, char* source, unsigned int elementSize) {
	for (ir::PTXOperand::Array::const_iterator i = instr.a.array.begin(); 
		i != instr.a.array.end(); ++i, source += elementSize) {
		switch (instr.type) {
			case ir::PTXOperand::b8:		// fall through
			case ir::PTXOperand::s8:		// fall through
			case ir::PTXOperand::u8:
				{
					ir::PTXU8 word = getRegAsU8(threadID, i->reg);
#if REPORT_ST
					report( "    Storing " << word << " to " 
						<< (int*)source );
#endif
					*((ir::PTXU8*)source) = word;
				}
				break;
			case ir::PTXOperand::b16:		// fall through
			case ir::PTXOperand::s16:		// fall through
			case ir::PTXOperand::u16:
				{
					ir::PTXU16 word = getRegAsU16(threadID, i->reg);
#if REPORT_ST
					report( "    Storing " << word << " to " 
						<< (int*)source );
#endif
					*((ir::PTXU16*)source) = word;
				}
				break;
		case ir::PTXOperand::f32:
			{
				ir::PTXF32 word = getRegAsF32(threadID, i->reg);
#if REPORT_ST
				report( "    Storing " << word << " to " 
					<< (int*)source << " - from register " << i->reg );
#endif
				*((ir::PTXF32*)source) = word;
			}
			break;
			case ir::PTXOperand::b32:		// fall through
			case ir::PTXOperand::s32:		// fall through
			case ir::PTXOperand::u32:
				{
					ir::PTXU32 word = getRegAsU32(threadID, i->reg);
#if REPORT_ST
					report( "    Storing " << word << " to " 
						<< (int*)source );
#endif
					*((ir::PTXU32*)source) = word;
				}
				break;
			case ir::PTXOperand::f64:		// fall through
			case ir::PTXOperand::b64:		// fall through
			case ir::PTXOperand::s64:		// fall through
			case ir::PTXOperand::u64:
				{
					ir::PTXU64 word = getRegAsU64(threadID, i->reg);
#if REPORT_ST
					report( "    Storing " << (void*)word << " to " 
						<< (int*)source );
#endif
					*((ir::PTXU64*)source) = word;
				}
				break;
			default:
				break;
		}
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_St(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	size_t elementSize = 0;

	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				elementSize = sizeof(ir::PTXU8);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				elementSize = sizeof(ir::PTXU16);
			}
			break;
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			char *source = 0;

			switch (instr.d.addressMode) {
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
				switch (instr.d.type) {
					case ir::PTXOperand::b8:		// fall through
					case ir::PTXOperand::s8:		// fall through
					case ir::PTXOperand::u8:
						{
							source += getRegAsU8(threadID, instr.d.reg );
						}
						break;
					case ir::PTXOperand::b16:		// fall through
					case ir::PTXOperand::s16:		// fall through
					case ir::PTXOperand::u16:
						{
							source += getRegAsU16(threadID, instr.d.reg );
						}
						break;
					case ir::PTXOperand::f32:		// fall through
					case ir::PTXOperand::b32:		// fall through
					case ir::PTXOperand::s32:		// fall through
					case ir::PTXOperand::u32:
						{
							source += getRegAsU32(threadID, instr.d.reg );
						}
						break;
					case ir::PTXOperand::f64:		// fall through
					case ir::PTXOperand::b64:		// fall through
					case ir::PTXOperand::s64:		// fall through
					case ir::PTXOperand::u64:
						{
							source += getRegAsU64(threadID, instr.d.reg );
						}
						break;
					default:
						break;
				}
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.d.imm_uint;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand", 
						context.PC, instr);
			}

			source += instr.d.offset;		

			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) {
			continue;
		}

		char *source = 0;

		switch (instr.d.addressMode) {
		case ir::PTXOperand::Register:
		case ir::PTXOperand::Indirect:
			switch (instr.d.type) {
				case ir::PTXOperand::b8:		// fall through
				case ir::PTXOperand::s8:		// fall through
				case ir::PTXOperand::u8:
					{
						source += getRegAsU8(threadID, instr.d.reg );
					}
					break;
				case ir::PTXOperand::b16:		// fall through
				case ir::PTXOperand::s16:		// fall through
				case ir::PTXOperand::u16:
					{
						source += getRegAsU16(threadID, instr.d.reg );
					}
					break;
				case ir::PTXOperand::f32:		// fall through
				case ir::PTXOperand::b32:		// fall through
				case ir::PTXOperand::s32:		// fall through
				case ir::PTXOperand::u32:
					{
						source += getRegAsU32(threadID, instr.d.reg );
					}
					break;
				case ir::PTXOperand::f64:		// fall through
				case ir::PTXOperand::b64:		// fall through
				case ir::PTXOperand::s64:		// fall through
				case ir::PTXOperand::u64:
					{
						source += getRegAsU64(threadID, instr.d.reg );
					}
					break;
				default:
					break;
			}
			break;
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			source += instr.d.imm_uint;
			break;
		default:
			throw RuntimeException(
				"unsupported address mode for source operand", 
					context.PC, instr);
		}

		source += instr.d.offset;
		
		switch (instr.addressSpace) {
			case ir::PTXInstruction::Param:
				{
					if (instr.d.isArgument) {
						source += kernel->scheduler->argumentMemory();
					}
					else {
						source += (ir::PTXU64) 
							functionCallStack.stackFramePointer(threadID);
					}
				}
				break;
				case ir::PTXInstruction::Generic: /* fall through */
				case ir::PTXInstruction::Global:			
				{	
				}
				break;
			case ir::PTXInstruction::Shared:
				{
					source += (ir::PTXU64) functionCallStack.sharedMemoryPointer();
				}
				break;
			case ir::PTXInstruction::Local:
				{
					if (instr.d.addressMode == ir::PTXOperand::Address &&
						instr.d.isGlobalLocal) {
						source += (ir::PTXU64) 
							functionCallStack.globalLocalMemoryPointer(
							threadID);
					}
					else {
						source += (ir::PTXU64) 
							functionCallStack.localMemoryPointer(threadID);
					}				
				}
				break;
			default:
				throw RuntimeException("unsupported address space", 
					context.PC, instr);
		}

		if (instr.a.vec == ir::PTXOperand::v1) {
			normalStore(threadID, instr, source);
		}
		else {
			vectorStore(threadID, instr, source, elementSize);
		}
	}

}

////////////////////////////////////////////////////////////////////////////////

/*!

*/		
void executive::CooperativeThreadArray::eval_Sub(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)), 
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			d = ftz(instr.modifier, sat(instr.modifier, a - b));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}	
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a), 
				b = operandAsF64(threadID, instr.b);
			d = a - b;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a), 
				b = operandAsS16(threadID, instr.b);
			d = a - b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a), 
				b = operandAsS32(threadID, instr.b);
				
			if (instr.modifier & ir::PTXInstruction::sat) {
				ir::PTXS64 la = a;
				ir::PTXS64 lb = b;

				ir::PTXS64 ld = la - lb;
				
				ld = max((ir::PTXS64)INT_MIN, ld);
				ld = min((ir::PTXS64)INT_MAX, ld);
				
				d = ld;
			}
			else if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXS32 carry = 0;
				
				hydrazine::add(d, carry, a, -b, carry);
				setRegAsS32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a - b;
			}
			
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a), 
				b = operandAsS64(threadID, instr.b);
			d = a - b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a), 
				b = operandAsU16(threadID, instr.b);
			d = a - b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a), 
				b = operandAsU32(threadID, instr.b);

			if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXU32 carry = 0;
				
				hydrazine::add(d, carry, a, -b, carry);
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a - b;
			}
			
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a), 
				b = operandAsU64(threadID, instr.b);
			d = a - b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_SubC(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {

	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d = 0,
				a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);

			ir::PTXU32 carry = getRegAsU32(threadID, instr.c.reg) - 1;

			hydrazine::add(d, carry, a, -b, carry);

			setRegAsU32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d = 0,
				a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);

			ir::PTXS32 carry = getRegAsS32(threadID, instr.c.reg) - 1;

			hydrazine::add(d, carry, a, -b, carry);

			setRegAsS32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsS32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!
	\brief load from surface memory
*/
void executive::CooperativeThreadArray::eval_Suld(CTAContext &context,
	const ir::PTXInstruction &instr) {
	switch (instr.geometry) {
	case ir::PTXInstruction::_1d:
	{
	}
		break;
	case ir::PTXInstruction::_2d:
	{
	
	}
		break;
	case ir::PTXInstruction::_3d:
	{
	
	}
		break;
	default:
		throw RuntimeException("suld - invalid geometry", context.PC, instr);
	}
	throw RuntimeException("suld - not implemented", context.PC, instr);
}

void executive::CooperativeThreadArray::eval_Sured(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("sured - not implemented", context.PC, instr);
}

/*!
	\brief store to surface memory
*/
void executive::CooperativeThreadArray::eval_Sust(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("sust - not implemented", context.PC, instr);
}

/*!

*/
void executive::CooperativeThreadArray::eval_TestP(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	const ir::PTXInstruction::FloatingPointMode mode = instr.floatingPointMode;

	switch(instr.type) {
	case ir::PTXOperand::f32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			bool d = false;
			ir::PTXF32 a = operandAsF32(threadID, instr.a);
			
			switch(mode)
			{
			case ir::PTXInstruction::Finite:
			{
				d = !hydrazine::isinf(a) && !hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::Infinite:
			{
				d = hydrazine::isinf(a);
			}
			break;
			case ir::PTXInstruction::Number:
			{
				d = !hydrazine::isnan(a);			
			}
			break;
			case ir::PTXInstruction::NotANumber:
			{
				d = hydrazine::isnan(a);			
			}
			break;
			case ir::PTXInstruction::Normal:
			{
				d = hydrazine::isnormal(a);
			}
			break;
			case ir::PTXInstruction::SubNormal:
			{
				d = issubnormal(a);
			}
			break;
			default: assertM(false, "Invalid floating point mode.");
			}
			
			setRegAsPredicate(threadID, instr.d.reg, d);
		}	
	}
	break;
	case ir::PTXOperand::f64:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			bool d = false;
			ir::PTXF64 a = operandAsF64(threadID, instr.a);
			
			switch(mode)
			{
			case ir::PTXInstruction::Finite:
			{
				d = !hydrazine::isinf(a) && !hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::Infinite:
			{
				d = hydrazine::isinf(a);
			}
			break;
			case ir::PTXInstruction::Number:
			{
				d = !hydrazine::isnan(a);			
			}
			break;
			case ir::PTXInstruction::NotANumber:
			{
				d = hydrazine::isnan(a);			
			}
			break;
			case ir::PTXInstruction::Normal:
			{
				d = hydrazine::isnormal(a);
			}
			break;
			case ir::PTXInstruction::SubNormal:
			{
				d = !hydrazine::isnormal(a) && !hydrazine::isnan(a) && !hydrazine::isinf(a);
			}
			break;
			default: assertM(false, "Invalid floating point mode.");
			}
			
			setRegAsPredicate(threadID, instr.d.reg, d);
		}	
	}
	break;
	default: throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Tex(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	const ir::Texture& texture = *context.kernel->textures[instr.a.reg];
	if (traceEvents) {
		currentEvent.memory_size = 4;
	}
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		switch (instr.geometry) {
			case ir::PTXInstruction::_1d:
				switch (instr.d.type) {
					case ir::PTXOperand::u32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXS32 c = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXF32 c = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;
						
					case ir::PTXOperand::s32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXS32 c = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXF32 c = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;
						
					case ir::PTXOperand::f32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXS32 c = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXF32 c = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;

					default:
						throw RuntimeException(
							"invalid texture destination type", 
							context.PC, instr);
				}
				break;

			case ir::PTXInstruction::_2d:
				switch (instr.d.type) {
					case ir::PTXOperand::u32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID, 
								instr.c.array[1].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID, 
								instr.c.array[1].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;
						
					case ir::PTXOperand::s32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID, 
								instr.c.array[1].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID, 
								instr.c.array[1].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;
						
					case ir::PTXOperand::f32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID, 
								instr.c.array[1].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID, 
								instr.c.array[1].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;

					default:
						throw RuntimeException(
							"invalid texture destination type", 
							context.PC, instr);
				}
				break;

			case ir::PTXInstruction::_3d:
				switch (instr.d.type) {
					case ir::PTXOperand::u32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID, 
								instr.c.array[1].reg);
							ir::PTXS32 c2 = getRegAsS32(threadID, 
								instr.c.array[2].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID, 
								instr.c.array[1].reg);
							ir::PTXF32 c2 = getRegAsF32(threadID, 
								instr.c.array[2].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;
						
					case ir::PTXOperand::s32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID, 
								instr.c.array[1].reg);
							ir::PTXS32 c2 = getRegAsS32(threadID, 
								instr.c.array[2].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID, 
								instr.c.array[1].reg);
							ir::PTXF32 c2 = getRegAsF32(threadID, 
								instr.c.array[2].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;
						
					case ir::PTXOperand::f32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID, 
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID, 
								instr.c.array[1].reg);
							ir::PTXS32 c2 = getRegAsS32(threadID, 
								instr.c.array[2].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID, 
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID, 
								instr.c.array[1].reg);
							ir::PTXF32 c2 = getRegAsF32(threadID, 
								instr.c.array[2].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type", 
								context.PC, instr);
						}
						break;

					default:
						throw RuntimeException(
							"invalid texture destination type", 
							context.PC, instr);
				}
				break;
			case ir::PTXInstruction::_cube:
				{
					if (texture.size.z != 6) {
						throw RuntimeException(
							"Invalid texture dimensions. Must have depth of 6.", context.PC, instr);
					}
					if (instr.type == ir::PTXOperand::f32) {
						switch (instr.d.type) {
							case ir::PTXOperand::f32:
							{
								assert(instr.c.array.size()==4);
								assert(instr.d.array.size()==4);
								ir::PTXF32 c0 = getRegAsF32(threadID, instr.c.array[0].reg);
								ir::PTXF32 c1 = getRegAsF32(threadID, instr.c.array[1].reg);
								ir::PTXF32 c2 = getRegAsF32(threadID, instr.c.array[2].reg);
								
								ir::PTXF32 d0 = tex::sampleCube<0,ir::PTXF32>(texture, c0, c1, c2);
								ir::PTXF32 d1 = tex::sampleCube<1,ir::PTXF32>(texture, c0, c1, c2);
								ir::PTXF32 d2 = tex::sampleCube<2,ir::PTXF32>(texture, c0, c1, c2);
								ir::PTXF32 d3 = tex::sampleCube<3,ir::PTXF32>(texture, c0, c1, c2);
								if (traceEvents) {
									tex::addresses(texture, c0, c1, c2,	currentEvent.memory_addresses);
								}
								setRegAsF32(threadID, instr.d.array[0].reg, d0);
								setRegAsF32(threadID, instr.d.array[1].reg, d1);
								setRegAsF32(threadID, instr.d.array[2].reg, d2);
								setRegAsF32(threadID, instr.d.array[3].reg, d3);
							}
							break;
						case ir::PTXOperand::u32:	// fall through for now
						case ir::PTXOperand::s32:	// fall through for now
						default:
							throw RuntimeException(
								"invalid texture destination type", 
								context.PC, instr);
						}
					}
					else {
						throw RuntimeException(
							"invalid texture source type", 
							context.PC, instr);
					}
				}
				break;
			default:
				throw RuntimeException("invalid texture geometry", 
					context.PC, instr);
				break;
		}
	}	
	trace();
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Trap(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	context.executionState = CTAContext::Trap;
}

void executive::CooperativeThreadArray::eval_Suq(CTAContext &context, const ir::PTXInstruction &instr) {
	// this instruction is identical to txq except for surfaces which we don't distinguish from textures
	eval_Txq(context, instr);
}

void executive::CooperativeThreadArray::eval_Txq(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	const ir::Texture& texture = *context.kernel->textures[instr.a.reg];
	switch (instr.surfaceQuery) {
	case ir::PTXInstruction::Width:	// fall through
	case ir::PTXInstruction::Height:	// fall through
	case ir::PTXInstruction::Depth:
	{
		int ir::Dim3::* dimensions[] = { &ir::Dim3::x, &ir::Dim3::y, &ir::Dim3::z };
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.size.*dimensions[instr.surfaceQuery - ir::PTXInstruction::Width]);
		}
	}
		break;
	case ir::PTXInstruction::ChannelDataType:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.type);
		}
	}
		break;
	case ir::PTXInstruction::ChannelOrder:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
		}
		throw RuntimeException("txq.channel_order - unimplemented query mode", context.PC, instr);
	}
		break;
	case ir::PTXInstruction::NormalizedCoordinates:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.normalize);
		}
	}
		break;
	case ir::PTXInstruction::SamplerFilterMode:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.interpolation);
		}
	}
		break;
	case ir::PTXInstruction::SamplerAddrMode0:	// fall through
	case ir::PTXInstruction::SamplerAddrMode1:	// fall through
	case ir::PTXInstruction::SamplerAddrMode2:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.addressMode[instr.surfaceQuery - ir::PTXInstruction::SamplerAddrMode0]);
		}
	}
		break;
	default:
		throw RuntimeException("txq - unexpected surface or texture query mode", context.PC, instr);
	}
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Vote(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	reconvergenceMechanism->eval_Vote(context, instr);
}

/*!

*/		
void executive::CooperativeThreadArray::eval_Xor(CTAContext &context, 
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			bool d, a = operandAsPredicate(threadID, instr.a), 
				b = operandAsPredicate(threadID, instr.b);
			d = b ^ a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a),
				b = operandAsB16(threadID, instr.b);
			d = a ^ b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a),
				b = operandAsB32(threadID, instr.b);
			d = a ^ b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			ir::PTXB64 d, a = operandAsB64(threadID, instr.a),
				b = operandAsB64(threadID, instr.b);
			d = a ^ b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			
			bool d, a = operandAsPredicate(threadID, instr.a), 
				b = operandAsPredicate(threadID, instr.b);
			d = a ^ b;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_cudaLaunchDevice(CTAContext &context,
	const ir::PTXInstruction &instr) {
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		// parameter 0 - the PC of the kernel being launched
		// parameter 1 - parameter buffer address
		// parameter 2 - kernel dim
		// parameter 3 - cta dim
		// parameter 4 - shared memory size
		// parameter 5 - stream

		ir::PTXU64 functionPointer =
			getFunctionParameter<ir::PTXU64>(threadID, instr, 0);
		ir::PTXU64 parameterBuffer  =
			getFunctionParameter<ir::PTXU64>(threadID, instr, 1);
		ir::Dim3   gridDim          =
			getFunctionParameter<ir::Dim3>(threadID, instr, 2);
		ir::Dim3   ctaDim           =
			getFunctionParameter<ir::Dim3>(threadID, instr, 3);
		ir::PTXU32 sharedMemory     =
			getFunctionParameter<ir::PTXU32>(threadID, instr, 4);
		ir::PTXU64 stream           =
			getFunctionParameter<ir::PTXU64>(threadID, instr, 5);
		
		kernel->scheduler->launch(functionPointer, parameterBuffer, gridDim,
			ctaDim, sharedMemory, stream);

		// return parameter 0 - the status of the call
		setFunctionParameter(threadID, instr, 0, (ir::PTXU32)0);
	}

	context.PC++;
}

void executive::CooperativeThreadArray::eval_cudaSynchronizeDevice(
	CTAContext &context,
	const ir::PTXInstruction &instr) {
	context.executionState = CTAContext::Barrier;
}

