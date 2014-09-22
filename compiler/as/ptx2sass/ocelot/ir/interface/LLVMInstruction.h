/*! \file LLVMInstruction.h
	\date Wednesday July 15, 2009
	\author Gregroy Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMInstruction class.
*/

#ifndef LLVM_INSTRUCTION_H_INCLUDED
#define LLVM_INSTRUCTION_H_INCLUDED

#include <ocelot/ir/interface/Instruction.h>
#include <vector>

namespace ir
{
	typedef bool LLVMI1;
	typedef char LLVMI8;
	typedef short LLVMI16;
	typedef int LLVMI32;
	typedef long long LLVMI64;
	typedef long long LLVMI128;
	
	typedef float LLVMF32;
	typedef double LLVMF64;
	typedef LLVMI128 LLVMF128;

	/*! \brief A class used to represent any LLVM Instruction */
	class LLVMInstruction : public Instruction
	{
		public:
			/*! \brief The opcode of the instruction */
			enum Opcode
			{
				Add,
				Alloca,
				And,
				Ashr,
				Atomicrmw,
				Bitcast,	
				Br,
				Call,
				Cmpxchg,
				Extractelement,
				Extractvalue,
				Fadd,
				Fcmp,
				Fdiv,
				Fmul,
				Fpext,	
				Fptosi,	
				Fptoui,
				Fptrunc,	
				Free,
				Frem,
				Fsub,
				Getelementptr,
				Icmp,
				Insertelement,
				Insertvalue,
				Inttoptr,
				Invoke,
				Load,
				Lshr,
				Malloc,
				Mul,
				Or,
				Phi,
				Ptrtoint,
				Ret,
				Sdiv,
				Select,
				Sext,
				Shl,
				Shufflevector,
				Sitofp,
				Srem,
				Store,
				Sub,
				Switch,
				Trunc,
				Udiv,
				Uitofp,	
				Unreachable,
				Unwind,
				Urem,
				VaArg,
				Xor,
				Zext,
				InvalidOpcode
			};
			
			/*! \brief Supported LLVM Types */
			enum DataType
			{
				I1,
				I8,
				I16,
				I32,
				I64,
				I128,
				F32,
				F64,
				F128,
				InvalidDataType
			};
			
			/*! \brief Floating Point Comparison */
			enum Comparison
			{
				False, // no comparison, always returns false
				Oeq, // ordered and equal
				Ogt, // ordered and greater than
				Oge, // ordered and greater than or equal
				Olt, // ordered and less than
				Ole, // ordered and less than or equal
				One, // ordered and not equal
				Ord, // ordered (no nans)
				Ueq, // unordered or equal
				Ugt, // unordered or greater than
				Uge, // unordered or greater than or equal
				Ult, // unordered or less than
				Ule, // unordered or less than or equal
				Une, // unordered or not equal
				Uno, // unordered (either nans)
				True, // no comparison, always returns true
				Eq, // Integer equal
				Ne, // Integer not equal
				Sgt, // Signed greater than
				Sge, // Signed greater equal
				Slt, // Signed less than
				Sle // Signed less or equal
			};
			
			/*! \brief Valid calling conventions */
			enum CallingConvention
			{
				CCallingConvention,
				FastCallingConvention,
				ColdCallingConvention,
				DefaultCallingConvention,
				InvalidCallingConvention
			};
			
			/*! \brief Valid Parameter Attributes */
			enum ParameterAttribute
			{
				ZeroExtend,
				SignExtend,
				InRegister,
				ByValue,
				StructureReturn,
				NoAlias,
				NoCapture,
				Nested,
				InvalidParameterAttribute
			};
			
			/*! \brief Type of atomic operation */
			enum AtomicOperation
			{
				AtomicXchg,
				AtomicAdd,
				AtomicSub,
				AtomicAnd,
				AtomicNand,
				AtomicOr,
				AtomicXor,
				AtomicMax,
				AtomicMin,
				AtomicUmax,
				AtomicUmin,
				InvalidAtomicOperation
			};
			
			/*! \brief Valid Function Attributes */
			enum FunctionAttribute
			{
				AlwaysInline = 1,
				NoInline = 2,
				OptimizeSize = 4,
				NoReturn = 8,
				NoUnwind = 16,
				ReadNone = 32,
				ReadOnly = 64,
				StackSmashingProtector = 128,
				StackSmashingProtectorRequired = 256,
				NoRedZone = 512,
				NoImplicitFloat = 1024,
				Naked = 2048
			};
			
			/*! \brief A class for an LLVM basic or derived type */
			class Type
			{
				public:
					/*! \brief All possible operand types */
					enum Category
					{
						Element, //! A single element of a base type
						Array, //! An array of elements
						Function, //! A function pointer
						Structure, //! An unconstrained ordered set of types
						PackedStructure, //! Structure with no padding
						Pointer, //! A pointer to a type
						Vector, //! A vector for use in SIMD instructions
						Opaque, //! An unknown type that has not been resolved
						InvalidCategory
					};
				
				public:
					/*! \brief An ordered set of types */
					typedef std::vector< Type > TypeVector;
			
				public:
					/*! \brief The other types contained */
					TypeVector members;
							
				public:
					/*! \brief The datatype of the Type */
					DataType type;
					/*! \brief The category of the Type */
					Category category;
					/*! \brief The label of the Type */
					std::string label;
					/*! \brief The vector width of the Type */
					LLVMI32 vector;
				
				public:
					/*! \brief The constructor sets the type and pointer flag */
					Type( DataType t = InvalidDataType, 
						Category c = InvalidCategory, LLVMI32 v = 1 );			
				
				public:			
					/*! \brief A parsable string representation of the Type */
					std::string toString() const;
			};
						
			/*! \brief The value of the operand if it is a constant */
			union Value
			{
				LLVMI1 i1;
				LLVMI8 i8;
				LLVMI16 i16;
				LLVMI32 i32;
				LLVMI64 i64;
				LLVMI128 i128;
				LLVMF32 f32;
				LLVMF64 f64;
				LLVMF128 f128;
			};
			
			/*! \brief A class for a basic LLVM Operand */
			class Operand
			{
				public:
					/*! \brief The name of the operand */
					std::string name;
					/*! \brief Is this a variable or a constant */
					bool constant;
					/*! \brief The type of the operand */
					Type type;
					/*! \brief The value of the operand if it is a constant */
					union
					{
						LLVMI1 i1;
						LLVMI8 i8;
						LLVMI16 i16;
						LLVMI32 i32;
						LLVMI64 i64;
						LLVMI128 i128;
						LLVMF32 f32;
						LLVMF64 f64;
						LLVMF128 f128;
					};
					
					/*! \brief A vector of values for a constant vector type */
					typedef std::vector< Value > ValueVector;
					
					/*! \brief If this is a constant vector type, the values */
					ValueVector values;
					
				public:
					/*! \brief The constructor sets the type and pointer flag */
					Operand( const std::string& n = std::string(), 
						const Type& t = Type() );
					/*! \brief The constructor sets the type and pointer flag */
					Operand( LLVMI64 value );
					/*! \brief The constructor sets the type and pointer flag */
					Operand( LLVMI32 value );
					/*! \brief The constructor sets the type and pointer flag */
					Operand( LLVMF32 value );
					/*! \brief The constructor sets the type and pointer flag */
					Operand( LLVMF64 value );
					/*! \brief The constructor sets the type and pointer flag */
					Operand( LLVMI1 value );
					/*! \brief Is this a valid operand ? */
					bool valid() const;
					/*! \brief Return a parsable represention of the Operand */
					std::string toString() const;
			};
			
			/*! \brief A parameter operand */
			class Parameter : public Operand
			{
				public:
					/*! \brief Attributes of the parameter */
					ParameterAttribute attribute;
					
				public:
					/*! \brief The constructor zeros out the attribute */
					Parameter();
					/*! \brief Construct from an operand */
					Parameter( const Operand& op );
					/*! \brief Assignment operator from an Operand */
					const Parameter& operator=( const Operand& op );
					/*! \brief Assignment operator from an Operand */
					const Parameter& operator=( const Parameter& op );
			};
			
			/*! \brief A vector of operands */
			typedef std::vector< Operand > OperandVector;
			
			/*! \brief A vector of Parameters */
			typedef std::vector< Parameter > ParameterVector;
			
		public:
		    /*! \brief The opcode of the instruction */
		    const Opcode opcode;
		
		public:
			/*! \brief Convert an opcode to a string parsable by LLVM */
			static std::string toString( Opcode o );
			/*! \brief Convert a datatype to a string parsable by LLVM */
			static std::string toString( DataType d );
			/*! \brief Convert a calling convention to a string */
			static std::string toString( CallingConvention cc );
			/*! \brief Convert a Parameter attribute to a string */
			static std::string toString( ParameterAttribute attribute );
			/*! \brief Convert an atomic operation to a string */
			static std::string toString( AtomicOperation op );
			/*! \brief Convert a Comparison to a string */
			static std::string toString( Comparison comp );
			/*! \brief Convert a series of function attributes to a string */
			static std::string functionAttributesToString( int attributes );
			/*! \brief Determine if a datatype is an int */
			static bool isInt( DataType d );
			/*! \brief Get an int with the specified number of bits */
			static DataType getIntOfSize( unsigned int bits );
			/*! \brief Get the number of bits in a DataType */
			static unsigned int bits( DataType type );
			
		public:
			/*! \brief Default constructor */
			LLVMInstruction( Opcode op = InvalidOpcode );
			
			/*! \brief Copy constructor to prevent reassignment of opcode */
			LLVMInstruction( const LLVMInstruction& i );
			
			/*! \brief Assignment operator to prevent modification of opcode */
			const LLVMInstruction& operator=( const LLVMInstruction& i );
								
		public:
			/*! \brief Return a pointer to a new Instruction */
			virtual Instruction* clone(bool copy=true) const = 0;
	
		public:
			virtual std::string toString() const = 0;
			virtual std::string valid() const = 0;
	};
	
	/*! \brief A generic 1 operand instruction */
	class LLVMUnaryInstruction : public LLVMInstruction
	{
		public:
			/*! \brief The destination operand */
			Operand d;
			
			/*! \brief The source operand */
			Operand a;
	
		public:
			/*! \brief Default constructor */
			LLVMUnaryInstruction( Opcode op = InvalidOpcode, 
				const Operand& d = Operand(), 
				const Operand& a = Operand() );
			
		public:
			virtual std::string toString() const;
			virtual std::string valid() const;

		public:
			virtual Instruction* clone(bool copy = true) const = 0;
	};
	
	/*! \brief A generic 2 operand instruction */
	class LLVMBinaryInstruction : public LLVMInstruction
	{
		public:
			/*! \brief The destination operand */
			Operand d;
			
			/*! \brief The first source operand */
			Operand a;

			/*! \brief The second source operand */
			Operand b;

		public:
			/*! \brief Default constructor */
			LLVMBinaryInstruction( Opcode op = InvalidOpcode, 
				const Operand& _d = Operand(), const Operand& _a = Operand(), 
				const Operand& _b = Operand() );
			
		public:
			virtual std::string toString() const;
			virtual std::string valid() const;

		public:
			virtual Instruction* clone(bool copy=true) const = 0;
	};
	
	/*! \brief A generic conversion instruction */
	class LLVMConversionInstruction : public LLVMUnaryInstruction
	{
		public:
			/*! \brief Default constructor */
			LLVMConversionInstruction( Opcode op = InvalidOpcode,
				const Operand& d = Operand(), 
				const Operand& a = Operand() );
			
		public:
			virtual std::string toString() const;
			virtual std::string valid() const;

		public:
			virtual Instruction* clone(bool copy=true) const = 0;
	};
	
	/*! \brief A generic comparison instruction */
	class LLVMComparisonInstruction : public LLVMBinaryInstruction
	{
		public:
			/*! \brief Comparison operator */
			Comparison comparison;
		
		public:
			/*! \brief Default constructor */
			LLVMComparisonInstruction( Opcode op = InvalidOpcode );
			
		public:
			virtual std::string toString() const;
			virtual std::string valid() const;

		public:
			virtual Instruction* clone(bool copy=true) const = 0;
	};
	
	/*! \brief The LLVM add instruction */
	class LLVMAdd : public LLVMBinaryInstruction
	{
		public:
			/*! \brief No unsigned wrap */
			LLVMI1 noUnsignedWrap;
			
			/*! \brief No signed wrap */
			LLVMI1 noSignedWrap;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMAdd( const Operand& d = Operand(), 
				const Operand& a = Operand(), const Operand& b = Operand(),
				bool nur = false, bool nsr = false );
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM alloca instruction */
	class LLVMAlloca : public LLVMInstruction
	{
		public:
			/*! \brief The number of elements allocated */
			LLVMI32 elements;
			
			/*! \brief The alignment of elements */
			LLVMI32 alignment;
			
			/*! \brief The destination operand */
			Operand d;
			
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMAlloca( LLVMI32 e = 1, LLVMI32 a = 1);
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM And instruction */
	class LLVMAnd : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMAnd();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM ashr instruction */
	class LLVMAshr : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMAshr();

		public:
			std::string valid() const;
		
		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM atomicrmw instruction */
	class LLVMAtomicrmw : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMAtomicrmw();

		public:
			AtomicOperation operation;

		public:
			std::string valid() const;
			std::string toString() const;
		
		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM bitcast instruction */
	class LLVMBitcast : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMBitcast();

		public:
			Instruction* clone(bool copy=true) const;
	};
		
	/*! \brief The LLVM br instruction */
	class LLVMBr : public LLVMInstruction
	{
		public:
			/*! \brief The condition operand or empty if none */
			Operand condition;
			
			/*! \brief The target label, default if condition is not specified */
			std::string iftrue;
			
			/*! \brief The iffalse label */
			std::string iffalse;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMBr();
			
		public:
			std::string toString() const;
			std::string valid() const;	

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM call instruction */
	class LLVMCall : public LLVMInstruction
	{
		public:
			/*! \brief Is this call eligible for tail call optimization? */
			LLVMI1 tail;
			
			/*! \brief The calling convention */
			CallingConvention convention;
			
			/*! \brief The return operand */
			Parameter d;
			
			/*! \brief The signature of the function pointer being called */
			std::string signature;
			
			/*!	\brief The function name called */
			std::string name;
			
			/*! \brief The set of parameters */
			ParameterVector parameters;
			
			/*! \brief Function attributes of the call */
			LLVMI32 functionAttributes;			
			
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMCall();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM cmpxchg instruction */
	class LLVMCmpxchg : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMCmpxchg();

		public:
			Operand c;

		public:
			std::string toString() const;
			std::string valid() const;
		
		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM extractelement instruction */
	class LLVMExtractelement : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMExtractelement();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM extractvalue instruction */
	class LLVMExtractvalue : public LLVMInstruction
	{
		public:
			/*! \brief A vector of indices */
			typedef std::vector< LLVMI32 > IndexVector;
	
		public:
			/*! \brief The destination operand */
			Operand d;
			
			/*! \brief The source operand, must be an aggregate type */
			Operand a;
			
			/*! \brief Indexes within the aggregate type */
			IndexVector indices;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMExtractvalue();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fadd instruction */
	class LLVMFadd : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFadd();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fcmp instruction */
	class LLVMFcmp : public LLVMComparisonInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFcmp();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fdiv instruction */
	class LLVMFdiv : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFdiv();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fmul instruction */
	class LLVMFmul : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFmul();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fpext instruction */
	class LLVMFpext : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFpext();

		public:
			Instruction* clone(bool copy=true) const;
	};
		
	/*! \brief The LLVM fptosi instruction */
	class LLVMFptosi : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFptosi();

		public:
			Instruction* clone(bool copy=true) const;
	};
		
	/*! \brief The LLVM fptoui instruction */
	class LLVMFptoui : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFptoui();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fptrunc instruction */
	class LLVMFptrunc : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFptrunc();

		public:
			Instruction* clone(bool copy=true) const;
	};
		
	/*! \brief The LLVM free instruction */
	class LLVMFree : public LLVMInstruction
	{
		public:
			/*! \brief The operand being freed */
			Operand a;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFree();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM frem instruction */
	class LLVMFrem : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFrem();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM fsub instruction */
	class LLVMFsub : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMFsub();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM getelementptr instruction */
	class LLVMGetelementptr : public LLVMInstruction
	{
		public:
			/*! \brief */
			typedef std::vector< LLVMI32 > IndexVector;
	
		public:
			/*! \brief The destination operand */
			Operand d;
			
			/*! \brief The source operand, must be an aggregate type */
			Operand a;
			
			/*! \brief Indexes within the aggregate type */
			IndexVector indices;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMGetelementptr();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM icmp instruction */
	class LLVMIcmp : public LLVMComparisonInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMIcmp();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM insertelement instruction */
	class LLVMInsertelement : public LLVMBinaryInstruction
	{
		public:
			/*! \brief Index operand */
			Operand c;
				
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMInsertelement();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM insertvalue instruction */
	class LLVMInsertvalue : public LLVMInstruction
	{
		public:
			/*! \brief */
			typedef std::vector< LLVMI32 > IndexVector;
	
		public:
			/*! \brief The destination operand */
			Operand d;
			
			/*! \brief The source operand, must be an aggregate type */
			Operand a;
			
			/*! \brief The source value to insert */
			Operand b;
			
			/*! \brief Indexes within the aggregate type */
			IndexVector indices;
			
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMInsertvalue();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM intotoptr instruction */
	class LLVMInttoptr : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMInttoptr();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM invoke instruction */
	class LLVMInvoke : public LLVMInstruction
	{
		public:
			/*! \brief The return operand */
			Parameter d;
			
			/*! \brief The calling convention */
			CallingConvention convention;
			
			/*! \brief The return parameter attributes */
			ParameterAttribute returnAttributes;
			
			/*! \brief The set of parameters */
			ParameterVector parameters;
			
			/*! \brief The name of the function being invoked */
			std::string name;
			
			/*! \brief The signature of the function being invoked */
			std::string signature;
			
			/*! \brief Function attributes of the call */
			LLVMI32 functionAttributes;
			
			/*! \brief The label reached when the callee returns */
			std::string tolabel;
			
			/*! \brief The label reached when the callee hits unwind */	
			std::string unwindlabel;

		public:
			/*! \brief The default constructor sets the opcode */
			LLVMInvoke();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM load instruction */
	class LLVMLoad : public LLVMUnaryInstruction
	{
		public:
			/*! \brief Is the load volatile */
			LLVMI1 isVolatile;
	
			/*! \brief The alignment requirement of the load */
			LLVMI32 alignment;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMLoad();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM lshr instruction */
	class LLVMLshr : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMLshr();
			
		public:
			std::string valid() const;
			
		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM malloc instruction */
	class LLVMMalloc : public LLVMInstruction
	{
		public:
			/*! \brief The number of elements allocated */
			Operand elements;
			
			/*! \brief The alignment of elements */
			LLVMI32 alignment;
			
			/*! \brief The destination operand */
			Operand d;

		public:
			/*! \brief The default constructor sets the opcode */
			LLVMMalloc();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM mul instruction */
	class LLVMMul : public LLVMBinaryInstruction
	{
		public:
			/*! \brief No unsigned wrap */
			LLVMI1 noUnsignedWrap;
			
			/*! \brief No signed wrap */
			LLVMI1 noSignedWrap;
		
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMMul();
			
		public:
			std::string toString() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM or instruction */
	class LLVMOr : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMOr();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM phi instruction */
	class LLVMPhi : public LLVMInstruction
	{
		public:
			/*! \brief Class for a combination of an operand and a label */
			class Node
			{
				public:
					/*! Operand */
					Operand operand;
					/*! Label of the BB that this operand comes from */
					std::string label;
					/*! Register id of the operand */
					Instruction::RegisterType reg; 
			};
			
			/*! \brief A vector of Nodes */
			typedef std::vector< Node > NodeVector;
		
		public:
			/*! \brief The destination operand */
			Operand d;
			/*! \brief The list of Phi Nodes */
			NodeVector nodes;
					
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMPhi();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM ptrtoint instruction */
	class LLVMPtrtoint : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMPtrtoint();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM Add instruction */
	class LLVMRet : public LLVMInstruction
	{
		public:
			/*! \brief The destination operand */
			Operand d;
		
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMRet();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM sdiv instruction */
	class LLVMSdiv : public LLVMBinaryInstruction
	{
		public:
			/*! \brief Should the division be gauranteed to be exact? */
			LLVMI1 exact;
			
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSdiv();
			
		public:
			std::string toString() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM select instruction */
	class LLVMSelect : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The condition that determines which operand to select */
			Operand condition;
	
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSelect();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM sext instruction */
	class LLVMSext : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSext();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM shl instruction */
	class LLVMShl : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMShl();
		
		public:
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM shufflevector instruction */
	class LLVMShufflevector : public LLVMBinaryInstruction
	{
		public:
			/*! \brief A Mask is a vector of indices */
			typedef std::vector< LLVMI32 > Mask;
	
		public:
			/*! \brief The shuffle mask */
			Mask mask;
		
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMShufflevector();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM sitofp instruction */
	class LLVMSitofp : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSitofp();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM srem instruction */
	class LLVMSrem : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSrem();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM store instruction */
	class LLVMStore: public LLVMUnaryInstruction
	{
		public:
			/*! \brief Is the load volatile */
			LLVMI1 isVolatile;
	
			/*! \brief The alignment requirement of the load */
			LLVMI32 alignment;
			
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMStore();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM sub instruction */
	class LLVMSub: public LLVMBinaryInstruction
	{
		public:
			/*! \brief No unsigned wrap */
			LLVMI1 noUnsignedWrap;
			
			/*! \brief No signed wrap */
			LLVMI1 noSignedWrap;
			
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSub();
			
		public:
			std::string toString() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM switch instruction */
	class LLVMSwitch : public LLVMInstruction
	{
		public:
			/*! \brief Class for a combination of an operand and a label */
			class Node
			{
				public:
					/*! Operand */
					Operand operand;
					/*! Label of the BB that this operand comes from */
					std::string label; 
			};
			
			/*! \brief A vector of Nodes */
			typedef std::vector< Node > NodeVector;
			
		public:
			/*! \brief Comparison value */
			Operand comparison;
			
			/*! \brief Default destination label */
			std::string defaultTarget;
			
			/*! \brief List of possible other destinations */
			NodeVector targets;
		
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMSwitch();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM trunc instruction */
	class LLVMTrunc : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMTrunc();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM udiv instruction */
	class LLVMUdiv : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMUdiv();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM uitofp instruction */
	class LLVMUitofp : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMUitofp();

		public:
			Instruction* clone(bool copy=true) const;
	};
		
	/*! \brief The LLVM unreachable instruction */
	class LLVMUnreachable : public LLVMInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMUnreachable();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM unwind instruction */
	class LLVMUnwind : public LLVMInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMUnwind();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM urem instruction */
	class LLVMUrem : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMUrem();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM va_arg instruction */
	class LLVMVaArg : public LLVMUnaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMVaArg();
			
		public:
			std::string toString() const;
			std::string valid() const;

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM xor instruction */
	class LLVMXor : public LLVMBinaryInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMXor();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
	/*! \brief The LLVM zext instruction */
	class LLVMZext : public LLVMConversionInstruction
	{
		public:
			/*! \brief The default constructor sets the opcode */
			LLVMZext();

		public:
			Instruction* clone(bool copy=true) const;
	};
	
}

#endif


