/*! \file CooperativeThreadArray.h

	\author Andrew Kerr <arkerr@gatech.edu>

	\date 5 February 2009

	\brief defines the state of each cooperative thread array with associated 
		code for emulating its execution
*/

#ifndef EXECUTIVE_COOPERATIVETHREADARRAY_H_INCLUDED
#define EXECUTIVE_COOPERATIVETHREADARRAY_H_INCLUDED

#include <ocelot/executive/interface/CTAContext.h>
#include <ocelot/executive/interface/ReconvergenceMechanism.h>
#include <ocelot/executive/interface/EmulatorCallStack.h>
#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/Kernel.h>
#include <ocelot/ir/interface/Texture.h>
#include <ocelot/trace/interface/TraceEvent.h>

namespace executive {

	class EmulatedKernel;

	/*! Defines state of cooperative thread array */
	class CooperativeThreadArray {
	public:
		typedef std::vector<CTABarrier> BarrierVector;
		typedef std::vector<ir::PTXU64> RegisterFile;

	public:
		/*! Constructs a cooperative thread array from an EmulatedKernel
				instance

			\param kernel pointer to EmulatedKernel to which this CTA belongs
			\param gridDim The dimensions of the kernel
			\param trace Enable trace generation
		*/
		CooperativeThreadArray(EmulatedKernel *kernel, 
			const ir::Dim3& gridDim, bool trace);
		
		CooperativeThreadArray(EmulatedKernel *kernel, 
			const ir::Dim3& gridDim, const ir::Dim3& ctaDim,
			unsigned int argumentMemorySize, unsigned int parameterMemorySize,
			unsigned int registerCount, unsigned int localMemorySize,
			unsigned int globalLocalMemorySize,
			unsigned int totalSharedMemorySize, bool trace);

		CooperativeThreadArray();

		/*! Destroys state associated with CTA */
		~CooperativeThreadArray();
		
		void reset();
		
		/*! initializes elements of the CTA */
		void initialize(const ir::Dim3 & block);

		/*! Initializes the CTA and executes the kernel for a given block */
		void execute(int PC = 0);
		
		/*! Jump to a specific PC for the current context */
		void jumpToPC(int PC);

		/*! Get the current PC of the executing CTA */
		int getPC() const;

		/* Get a snapshot of the current register file */
		RegisterFile getCurrentRegisterFile() const;

		/*! gets the active context of the cooperative thread array */
		CTAContext& getActiveContext();

		/*! gets the active context of the cooperative thread array */
		const CTAContext& getActiveContext() const;

		/*! gets the context state of the currently active context */
		CTAContext::ExecutionState getExecutionState() const;
		
		/*! sets the context state of the currently active context */
		void setExecutionState(CTAContext::ExecutionState state);
		
	protected:
		
		/*! finishes execution of the CTA */
		void finalize();

	public:
		/*! Dimensions of the cooperative thread array */
		ir::Dim3 blockDim;

        /*! Dimensions of the kernel */
        ir::Dim3 gridDim;

		/*!
			Number of threads in CTA 
				(equal to blockDim.x * blockDim.y * blockDim.z)
		*/
		int threadCount;

		/*! Pointer to EmulatedKernel instance that this CTA is executing */
		EmulatedKernel *kernel;

		/*! ID of block implemented by this CooperativeThreadArray instance */
		ir::Dim3 blockId;

		/*! Function call stack */
		EmulatorCallStack functionCallStack;
		
		/*! Vector of named barriers */
		BarrierVector barriers;

		/*! \brief abstraction for reconvergence mechanism */
		ReconvergenceMechanism *reconvergenceMechanism;
		
		/*! Counter incremented 4 times per instruction */
		ir::PTXU64 clock;

		/*! Flag to enable or disable tracing of events */
		bool traceEvents;

		/*! Number of dynamic instructions executed */
		int counter;

		/*! An object used to trace execution of the CooperativeThreadArray */
		trace::TraceEvent currentEvent;

	protected:
		// internal functions for execution

		/*! Gets current instruction */
		const ir::PTXInstruction& currentInstruction(CTAContext& context);

		/*! Gets special value */
		ir::PTXU32 getSpecialValue(const int threadId,
			const ir::PTXOperand::SpecialRegister,
			const ir::PTXOperand::VectorIndex) const;
		
	protected:
		// execution helper functions
		
		// Implement saturation
		ir::PTXF32 sat(int modifier, ir::PTXF32 f);
		ir::PTXF64 sat(int modifier, ir::PTXF64 f);
		
		// Set the trace event
		void trace();

		/*!
			\brief invokes TraceGenerator::postEvent() on all trace generators
		*/		
		void postTrace();

	public:
		// Register accessors

		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXU8 getRegAsU8(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXU16 getRegAsU16(int threadID, ir::PTXOperand::RegisterType reg);
				
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXU32 getRegAsU32(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXU64 getRegAsU64(int threadID, ir::PTXOperand::RegisterType reg);

		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXS8 getRegAsS8(int threadID, ir::PTXOperand::RegisterType reg);
				
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXS16 getRegAsS16(int threadID, ir::PTXOperand::RegisterType reg);
				
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXS32 getRegAsS32(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXS64 getRegAsS64(int threadID, ir::PTXOperand::RegisterType reg);

		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXF32 getRegAsF32(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXF64 getRegAsF64(int threadID, ir::PTXOperand::RegisterType reg);

		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXB8 getRegAsB8(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXB16 getRegAsB16(int threadID, ir::PTXOperand::RegisterType reg);
				
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXB32 getRegAsB32(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		ir::PTXB64 getRegAsB64(int threadID, ir::PTXOperand::RegisterType reg);
		
		/*!
			Gets a register value
			
			\param threadID ID of the active thread
			\param reg index of register
		*/
		bool getRegAsPredicate(int threadID, ir::PTXOperand::RegisterType reg);
		
	public:

		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsU8(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXU8 value);
			
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsU16(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXU16 value);
				
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsU32(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXU32 value);
		
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsU64(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXU64 value);

		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsS8(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXS8 value);
					
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsS16(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXS16 value);
				
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsS32(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXS32 value);
		
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsS64(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXS64 value);

		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsF32(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXF32 value);
		
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsF64(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXF64 value);

		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsB8(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXB8 value);

		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsB16(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXB16 value);
				
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsB32(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXB32 value);
		
		/*!
			Sets a register value 

			\param threadID ID of the active thread
			\reg register index
		*/
		void setRegAsB64(int threadID, ir::PTXOperand::RegisterType reg, 
			ir::PTXB64 value);
		
		/*!
			Sets a register value
			
			\param threadID ID of the active thread
			\param reg index of register
			\param value value of predicate register
		*/
		void setRegAsPredicate(int threadID, ir::PTXOperand::RegisterType reg, 
			bool value);
	
	public:
		ir::PTXU8 operandAsU8(int, const ir::PTXOperand &);
		ir::PTXU16 operandAsU16(int, const ir::PTXOperand &);
		ir::PTXU32 operandAsU32(int, const ir::PTXOperand &);
		ir::PTXU64 operandAsU64(int, const ir::PTXOperand &);

		ir::PTXS8 operandAsS8(int, const ir::PTXOperand &);
		ir::PTXS16 operandAsS16(int, const ir::PTXOperand &);
		ir::PTXS32 operandAsS32(int, const ir::PTXOperand &);
		ir::PTXS64 operandAsS64(int, const ir::PTXOperand &);

		ir::PTXF32 operandAsF32(int, const ir::PTXOperand &);
		ir::PTXF64 operandAsF64(int, const ir::PTXOperand &);

		ir::PTXB8 operandAsB8(int, const ir::PTXOperand &);
		ir::PTXB16 operandAsB16(int, const ir::PTXOperand &);
		ir::PTXB32 operandAsB32(int, const ir::PTXOperand &);
		ir::PTXB64 operandAsB64(int, const ir::PTXOperand &);

		bool operandAsPredicate(int, const ir::PTXOperand&);

	private:
		template<typename T>
		T getFunctionParameter(int threadID,
			const ir::PTXInstruction& i, int index);
		template<typename T>
		void setFunctionParameter(int threadID,
			const ir::PTXInstruction& i, int index, T value);

	private:
		void normalStore(int, const ir::PTXInstruction &, char*);
		void vectorStore(int, const ir::PTXInstruction &, char*, unsigned int);
		void normalLoad(int, const ir::PTXInstruction &, const char*);
		void vectorLoad(int, const ir::PTXInstruction &, const char*, 
			unsigned int);

	public:
		/*Handlers for each instruction */
		void eval_Abs(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Add(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_AddC(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_And(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Atom(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Bar(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Bfi(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Bfind(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Bfe(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Bra(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Brev(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Brkpt(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Call(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Clz(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_CNot(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_CopySign(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Cos(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Cvt(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Cvta(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Div(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Ex2(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Exit(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Fma(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Isspacep(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Ld(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Ldu(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Lg2(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Mad24(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Mad(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Max(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Membar(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Min(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Mov(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Mul24(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Mul(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Neg(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Not(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Or(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Pmevent(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Popc(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Prefetch(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Prefetchu(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Prmt(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Rcp(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Red(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Rem(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Ret(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Rsqrt(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Sad(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_SelP(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_SetP(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Set(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Shl(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Shr(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Sin(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_SlCt(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Sqrt(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_St(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Sub(CTAContext &context, const ir::PTXInstruction &instr);	
		void eval_SubC(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Suld(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Sured(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Sust(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Suq(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_TestP(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Tex(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Trap(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Txq(CTAContext &context, const ir::PTXInstruction &instr);
		void eval_Vote(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Xor(CTAContext &context, const ir::PTXInstruction &instr);		
		void eval_Reconverge(CTAContext &context,
			const ir::PTXInstruction &instr);	

		// CNP support
		void eval_cudaLaunchDevice(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_cudaSynchronizeDevice(CTAContext &context,
			const ir::PTXInstruction &instr);	
	

	protected:
		
		void eval_Mov_reg(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Mov_sreg(CTAContext &context, 
			const ir::PTXInstruction &instr);
		void eval_Mov_imm(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Mov_indirect(CTAContext &context, 
			const ir::PTXInstruction &instr);
		void eval_Mov_addr(CTAContext &context,
			const ir::PTXInstruction &instr);
		void eval_Mov_func(CTAContext &context,
			const ir::PTXInstruction &instr);

		void copyArgument(const ir::PTXOperand& s, CTAContext& context);

	private:
		// CNP Support
		ir::PTXU64 getNewParameterBuffer(ir::PTXU64 alignment, ir::PTXU64 size);
		void freeParameterBuffer(ir::PTXU64 );
	};

}

#endif

