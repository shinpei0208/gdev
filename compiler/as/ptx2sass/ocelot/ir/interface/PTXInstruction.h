/*! \file PTXInstruction.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief base class for all instructions
*/

#ifndef IR_PTXINSTRUCTION_H_INCLUDED
#define IR_PTXINSTRUCTION_H_INCLUDED

#include <ocelot/ir/interface/Instruction.h>
#include <ocelot/ir/interface/PTXOperand.h>

namespace ir {

	class PTXInstruction: public Instruction {
	public:
		/*! Hierarchy Level */
		enum Level {
			CtaLevel,
			GlobalLevel,
			SystemLevel,
			Level_Invalid
		};

		/*!	List of opcodes for PTX instructions */
		enum Opcode {
			Abs = 0,
			Add,
			AddC,
			And,
			Atom,
			Bar,
			Bfe,
			Bfi,
			Bfind,
			Bra,
			Brev,
			Brkpt,
			Call,
			Clz,
			CNot,
			CopySign,
			Cos,
			Cvt,
			Cvta,
			Div,
			Ex2,
			Exit,
			Fma,
			Isspacep,
			Ld,
			Ldu,
			Lg2,
			Mad24,
			Mad,
			MadC,
			Max,
			Membar,
			Min,
			Mov,
			Mul24,
			Mul,
			Neg,
			Not,
			Or,
			Pmevent,
			Popc,
			Prefetch,
			Prefetchu,
			Prmt,
			Rcp,
			Red,
			Rem,
			Ret,
			Rsqrt,
			Sad,
			SelP,
			Set,
			SetP,
			Shfl,
			Shl,
			Shr,
			Sin,
			SlCt,
			Sqrt,
			St,
			Sub,
			SubC,
			Suld,
			Sured,
			Sust,
			Suq,
			TestP,
			Tex,
			Tld4,
			Txq,
			Trap,
			Vabsdiff,
			Vadd,
			Vmad,
			Vmax,
			Vmin,
			Vset,
			Vshl,
			Vshr,
			Vsub,
			Vote,
			Xor,

			// Special instructions inserted by the analysis procedures
			Reconverge,
			Phi,
			Nop,
			Invalid_Opcode
		};

		/*!
			Modifiers for integer and floating-point instructions. The
			FP rounding-modes are mutually exclusive but compatible with
			sat.
		*/
		enum Modifier {
			hi = 1,			//< 
			lo = 2,			//<
			wide = 4,		//<
			sat = 8,		//< saturation modifier	
			rn = 32,		//< mantissa LSB rounds to nearest even
			rz = 64,		//< mantissa LSB rounds toward zero
			rm = 128,		//< mantissa LSB rounds toward negative infty
			rp = 256,		//< mantissa LSB rounds toward positive infty
			rni = 512,	//< round to nearest integer, choosing even integer if source is equidistant between two integers.
			rzi = 1024,	//< round to nearest integer in the direction of zero
			rmi = 2048,	//< round to nearest integer in direction of negative infinity
			rpi = 4096,	//< round to nearest integer in direction of positive infinity
			approx = 8192,//< identify an approximate instruction
			ftz = 16384,	//< flush to zero
			full = 32768,	//< full division
			Modifier_invalid = 0
		};
			
		enum CarryFlag {
			None = 0,
			CC = 1
		};
		
		enum Volatility {
			Nonvolatile = 0,
			Volatile
		};
	
		enum AddressSpace {
			AddressSpace_Invalid = 0,
			Const,
			Global,
			Local,
			Param,
			Shared,
			Texture,
			Generic
		};

		enum AtomicOperation {
			AtomicAnd,
			AtomicOr,
			AtomicXor,
			AtomicCas,
			AtomicExch,
			AtomicAdd,
			AtomicInc,
			AtomicDec, 
			AtomicMin,
			AtomicMax,
			AtomicOperation_Invalid
		};
		
		enum BarrierOperation {
			BarSync,
			BarArrive,
			BarReduction,
			BarrierOperation_invalid
		};
		
		enum ReductionOperation {
			ReductionAnd,
			ReductionXor,
			ReductionOr,
			ReductionAdd,
			ReductionInc,
			ReductionDec,
			ReductionMin,
			ReductionMax,
			ReductionPopc,
			ReductionOperation_Invalid
		};
		
		enum CacheOperation {
			Ca = 0,
			Cv = 1,
			Cg = 2,
			Cs = 3,
			Nc = 4,
			Wb = 0,
			Wt = 1,
			CacheOperation_Invalid
		};
		
		enum CacheLevel {
			L1,
			L2,
			CacheLevel_invalid
		};
		
		enum ClampOperation {
			TrapOOB,
			Clamp,
			Zero,
			Mirror,
			ClampOperation_Invalid
		};

		enum ColorComponent {
			red,
			green,
			blue,
			alpha,
			ColorComponent_Invalid
		};

		/*! comparison operator */
		enum CmpOp {
			Eq,
			Ne,
			Lt,
			Le,
			Gt,
			Ge,
			Lo,
			Ls,
			Hi,
			Hs,
			Equ,
			Neu,
			Ltu,
			Leu,
			Gtu,
			Geu,
			Num,
			Nan,
			CmpOp_Invalid
		};
		
		enum PermuteMode {
			DefaultPermute,
			ForwardFourExtract,
			BackwardFourExtract,
			ReplicateEight,
			EdgeClampLeft,
			EdgeClampRight,
			ReplicateSixteen
		};
			
		enum FloatingPointMode {
			Finite,
			Infinite,
			Number,
			NotANumber,
			Normal,
			SubNormal,
			FloatingPointMode_Invalid
		};
		
		enum FormatMode {
			Unformatted,
			Formatted,
			FormatMode_Invalid
		};
		
		/*! Vector operation */
		typedef PTXOperand::Vec Vec;
		
		/*! boolean operator */
		enum BoolOp {
			BoolAnd,
			BoolOr,
			BoolXor,
			BoolNop,
			BoolOp_Invalid
		};
		
		/*! geometry for textures */
		enum Geometry {
			_1d = 1,
			_2d = 2,
			_3d = 3,
			_a1d = 4,
			_a2d = 5,
			_cube = 6,
			_acube = 7,
			Geometry_Invalid = 0
		};

		enum SurfaceQuery {
			Width,
			Height,
			Depth,
			ChannelDataType,
			ChannelOrder,
			NormalizedCoordinates,
			SamplerFilterMode,
			SamplerAddrMode0,
			SamplerAddrMode1,
			SamplerAddrMode2,
			SurfaceQuery_Invalid
		};

		enum VoteMode {
			All,
			Any,
			Uni,
			Ballot,
			VoteMode_Invalid
		};
		
		enum ShuffleMode {
			Up,
			Down,
			Bfly,
			Idx,
			ShuffleMode_Invalid
		};
		
	public:
		static std::string toString( Level );
		static std::string toString( CacheLevel cache );
		static std::string toStringLoad( CacheOperation op );
		static std::string toStringStore( CacheOperation op );
		static std::string toString( PermuteMode );
		static std::string toString( FloatingPointMode );
		static std::string toString( Vec );
		static std::string toString( AddressSpace );
		static std::string toString( AtomicOperation );
		static std::string toString( BarrierOperation );
		static std::string toString( ReductionOperation );
		static std::string toString( SurfaceQuery );
		static std::string toString( FormatMode );
		static std::string toString( ClampOperation );
		static std::string toString( CmpOp );
		static std::string toString( BoolOp );
		static std::string roundingMode( Modifier );
		static std::string toString( Modifier );
		static std::string toString( Geometry );
		static std::string modifierString( unsigned int, CarryFlag = None );
		static std::string toString( VoteMode );
		static std::string toString( ColorComponent );
		static std::string toString( ShuffleMode );
		static std::string toString( Opcode );
		static bool isPt( const PTXOperand& );

	public:
		PTXInstruction( Opcode op = Nop, const PTXOperand& d = PTXOperand(), 
			const PTXOperand& a = PTXOperand(), 
			const PTXOperand& b = PTXOperand(), 
			const PTXOperand& c = PTXOperand() );
		~PTXInstruction();

		bool operator==( const PTXInstruction& ) const;
		
		/*! Is this a valid instruction?
			
			return null string if valid, otherwise a description of why 
				not.
		*/
		std::string valid() const;

		/*! Returns the guard predicate representation for the instruction */
		std::string guard() const;

		/*! Returns a parsable string representation of the instruction */
		std::string toString() const;

		/*! \brief Clone the instruction */
		Instruction* clone( bool copy = true ) const;

	public:
		/*! \brief Is the instruction a branch */
		bool isBranch() const;
		/*! \brief Is the instruction a call */
		bool isCall() const;
		/*! \brief Is the instruction a load */
		bool isLoad() const;
		/*! \brief Is the instruction a store? */
		bool isStore() const;
		/*! \brief Does the instruction accept an address as an operand */
		bool mayHaveAddressableOperand() const;
		/*! \brief Does the instruction write to a relaxed type? */
		bool mayHaveRelaxedTypeDestination() const;
		/*! \brief Can the instruction affect state other than destinations? */
		bool hasSideEffects() const;
		/*! \brief Can the instruction observe state produced by
			other threads? */
		bool canObserveSideEffects() const;
		/*! \brief Does the instruction trigger a memory operation */
		bool isMemoryInstruction() const;
		/*! \brief Can the instruction exit the kernel/function */
		bool isExit() const;

	public:
		/*! Opcode of PTX instruction */
		Opcode opcode;

		/*! indicates data type of instruction */
		PTXOperand::DataType type;

		/*! Flag containing one or more floating-point modifiers */
		unsigned int modifier;

		union {
			/*! Shuffle mode */
			ShuffleMode shuffleMode;
		
			/*! Comparison operator */
			CmpOp comparisonOperator;

			/*! For load and store, indicates which addressing mode to use */
			AddressSpace addressSpace;
			
			/*! For membar, the visibility level in the thread hierarchy */
			Level level;
			
			/*! Shift amount flag for bfind instructions */
			bool shiftAmount;
			
			/*! Permute mode for prmt instructions */
			PermuteMode permuteMode;
			
			
			/*! For txq and suq instruction, specifies attributes */
			SurfaceQuery surfaceQuery;
			
			/*! For sust and suld instructions, indicates whether to store 
				unformatted binary datay or formatted store of a vector
				of 32-bit data */
			FormatMode formatMode;
			
			/*! Indicates which type of bar. instruction should be used */
			BarrierOperation barrierOperation;
			
			/*! For tld4 instructions, the color component */
			ColorComponent colorComponent;
			
		};
		
		/*! For call instructions, indicates a tail call */
		bool tailCall;
	
		/*! If the instruction is predicated, the guard */
		PTXOperand pg;
				
		/*! Second destination register for SetP, otherwise unused */
		PTXOperand pq;
		
		/*! Indicates whether target or source is a vector or scalar */
		Vec vec;

		union {

			/*! If instruction type is atomic, select this atomic operation */
			AtomicOperation atomicOperation;
			
			/*! If the instruction is a reduction, select this operation */
			ReductionOperation reductionOperation;
			
			/* If instruction type is vote, specifies the mode of voting */
			VoteMode vote;
			
			/* For TestP instructions, specifies the floating point mode */
			FloatingPointMode floatingPointMode;
			
			/* If instruction is a branch, is it .uni */
			bool uni;

			/*! Boolean operator */
			BoolOp booleanOperator;

			/*! Indicates whether the target address space is volatile */
			Volatility volatility;
						
			/*! Is this a divide full instruction? */
			bool divideFull;
			
			/*! If cvta instruction, indicates whether destination is 
				generic address or if source is generic address - true if 
				segmented address space, false if generic */
			bool toAddrSpace;
			
			/*! If the instruction updates the CC, what is the CC register */
			PTXOperand::RegisterType cc;
			
			/*! cache level */
			CacheLevel cacheLevel;
			
		};
		
		union {
		
			/*! Geometry if this is a texture or surface instruction */
			Geometry geometry;
			
			/*! indicates how loads, stores, and prefetches should take place */
			CacheOperation cacheOperation;
			
		};
		
		union {
		
			/*! optionally writes carry-out value to condition code register */
			CarryFlag carry;
		
			/*! how to handle out-of-bounds accesses */
			ClampOperation clamp;
			
		};

		/*! Destination operand */
		PTXOperand d;

		/*! Source operand a */
		PTXOperand a;

		/*! Source operand b */
		PTXOperand b;

		/*! Source operand c */
		PTXOperand c;

		/*  Runtime annotations 
			
			The following members are used to annotate the instruction 
				at analysis time for use at runtime
		*/
	public:
	
		union {
		
			/*! \brief Index of post dominator instruction at which possibly 
				divergent branches reconverge */
			int reconvergeInstruction;
			/*! \brief If this is a branch, is a check for re-convergence with
				threads waiting at the target necessary */
			bool needsReconvergenceCheck;
			
		};

		union {
		
			/*! \brief Branch target instruction index */
			int branchTargetInstruction;
			/*! \brief Context switch reentry point */
			int reentryPoint;
			/*! \brief Is this a kernel argument in the parameter space? */
			bool isArgument;
			/*! \brief Get or set the active mask */
			bool getActiveMask;
			
		};
		
		/*!	The following are used for debugging information at runtime. */
	public:
		/*! \brief The index of the statement that this instruction was 
			created from */
		unsigned int statementIndex;
		/*! \brief The program counter of the instruction */
		unsigned int pc;

		/*!	The following are used for debugging meta-data. */
	public:
		/*! \brief Meta-data attached to the instruction */
		std::string metadata;
	};

}

#endif

