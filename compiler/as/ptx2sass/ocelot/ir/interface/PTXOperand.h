/*! \file PTXOperand.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief internal representation of a PTX operand
*/

#ifndef IR_PTXOPERAND_H_INCLUDED
#define IR_PTXOPERAND_H_INCLUDED

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <ocelot/ir/interface/Instruction.h>

namespace ir {

	typedef uint8_t PTXU8;
	typedef uint16_t PTXU16;
	typedef uint32_t PTXU32;
	typedef uint64_t PTXU64;
	
	typedef int8_t PTXS8;
	typedef int16_t PTXS16;
	typedef int32_t PTXS32;
	typedef int64_t PTXS64;
	
	typedef float PTXF32;
	typedef double PTXF64;
	
	typedef PTXU8   PTXB8;
	typedef PTXU16  PTXB16;
	typedef PTXU32  PTXB32;
	typedef PTXU64  PTXB64;

	class PTXOperand {
	public:
		//! addressing mode of operand
		enum AddressMode {
			Register,			//! use as register variable
			Indirect,			//! indirect access
			Immediate,			//! treat as immediate value
			Address,			//! treat as addressable variable
			Label,				//! operand is a label
			Special,			//! special register
			ArgumentList,       //! treat as argument list for function call
			FunctionName,       //! operand is a function name
			BitBucket,			//! bit bucket register
			Invalid
		};

		/*! Type specifiers for instructions */
		enum DataType {
			TypeSpecifier_invalid = 0,
			s8,
			s16,
			s32,
			s64,
			u8,
			u16,
			u32,
			u64,
			f16,
			f32,
			f64,
			b8,
			b16,
			b32,
			b64,
			pred
		};

		/*!	Special register names */
		enum SpecialRegister {
			tid,
			ntid,
			laneId,
			warpId,
			nwarpId,
			warpSize,
			ctaId,
			nctaId,
			smId,
			nsmId,
			gridId,
			clock,
			clock64,
			lanemask_eq,
			lanemask_le,
			lanemask_lt,
			lanemask_ge,
			lanemask_gt,
			pm0,
			pm1,
			pm2,
			pm3,
			envreg0,
			envreg1,
			envreg2,
			envreg3,
			envreg4,
			envreg5,
			envreg6,
			envreg7,
			envreg8,
			envreg9,
			envreg10,
			envreg11,
			envreg12,
			envreg13,
			envreg14,
			envreg15,
			envreg16,
			envreg17,
			envreg18,
			envreg19,
			envreg20,
			envreg21,
			envreg22,
			envreg23,
			envreg24,
			envreg25,
			envreg26,
			envreg27,
			envreg28,
			envreg29,
			envreg30,
			envreg31,
			SpecialRegister_invalid
		};
		
		enum PredicateCondition {
			Pred,		//< instruction executes if predicate is true
			InvPred,	//< instruction executes if predicate is false
			PT,			//< predicate is always true
			nPT			//< predicate is always false
		};
	
		enum Vec {
			v1 = 1, 			//< scalar
			v2 = 2,				//< vector2
			v4 = 4				//< vector4
		};
		
		enum VectorIndex {
			iAll = 0, //! Refers to the complete vector
			ix = 1, //! Only refers to the x index of the vector
			iy = 2, //! Only refers to the y index of the vector
			iz = 3, //! Only refers to the z index of the vector
			iw = 4 //! Only refers to the w index of the vector
		};

		typedef std::vector<PTXOperand> Array;

		typedef Instruction::RegisterType RegisterType;

	public:
		static std::string toString(VectorIndex);
		static std::string toString(Vec);
		static std::string toString(DataType);
		static std::string toString(SpecialRegister);
		static std::string toString(AddressMode);
		static std::string toString(DataType, RegisterType);
		static bool isFloat(DataType);
		static bool isInt(DataType);
		static bool isSigned(DataType);
		static unsigned int bytes(DataType);
		static bool valid(DataType, DataType);
		static bool relaxedValid(DataType instructionType, DataType);
		static long long unsigned int maxInt(DataType type);
		static long long unsigned int minInt(DataType type);
		
	public:
		PTXOperand();
		PTXOperand(SpecialRegister r, VectorIndex i = iAll, DataType t = u32);
		PTXOperand(const std::string& label);
		PTXOperand(AddressMode m, DataType t, RegisterType r = 0, 
			int o = 0, Vec v = v1);
		PTXOperand(AddressMode m, DataType t, const std::string& identifier, 
			int o = 0, Vec v = v1);
		PTXOperand(AddressMode m, const std::string& identifier);
		PTXOperand(PredicateCondition condition);
		
		template<typename T>
		PTXOperand(T v, DataType t);
		template<typename T>
		explicit PTXOperand(T v);
		
		~PTXOperand();

		std::string toString() const;
		std::string registerName() const;
		unsigned int bytes() const;
		bool isRegister() const;
		bool isVector() const;

		//! identifier of operand
		std::string identifier;
		
		//! addressing mode of operand
		AddressMode addressMode;

		//! data type for PTX instruction
		DataType type;
		DataType relaxedType;

		//!	offset when used with an indirect addressing mode
		union {
			int offset;
			VectorIndex vIndex;
			unsigned int registerCount;
		};
		
		//! immediate-mode value of operand
		union {
			long long unsigned int imm_uint;
			long long int imm_int;
			double imm_float;
			float  imm_single;
			PredicateCondition condition;
			SpecialRegister special;
			unsigned int localMemorySize;
		};

		union {
			/*! Identifier for register */
			RegisterType reg;
			bool isArgument;
			bool isGlobalLocal;
			unsigned int sharedMemorySize;
		};
		
		union {
			//! Indicates whether target or source is a vector or scalar
			Vec vec;
			unsigned int stackMemorySize;
		};
		
		//! Array if this is a vector
		Array array;
		
	};

}

namespace ir {
	template<typename T>
	PTXOperand::PTXOperand(T v, DataType t) : addressMode(Immediate), type(t),
		relaxedType(TypeSpecifier_invalid),
		offset(0), imm_uint(v), reg(0), vec(v1) {}
	
	template<typename T>
	PTXOperand::PTXOperand(T v) : addressMode(Immediate), type(u64),
		relaxedType(TypeSpecifier_invalid),
		offset(0), imm_uint(v), reg(0), vec(v1) {}

	template<> inline
	PTXOperand::PTXOperand(const char* v) : identifier(v), 
		addressMode(Label), type(TypeSpecifier_invalid),
		relaxedType(TypeSpecifier_invalid),
		offset(0), condition(Pred), reg(0), vec(v1) {
	}

	template<> inline
	PTXOperand::PTXOperand(float v) : addressMode(Immediate), type(f32),
		relaxedType(TypeSpecifier_invalid),
		offset(0), imm_single(v), reg(0), vec(v1) {
	}
	
	template<> inline
	PTXOperand::PTXOperand(double v) : addressMode(Immediate),
		type(f64), relaxedType(TypeSpecifier_invalid),
		offset(0), imm_float(v), reg(0), vec(v1) {
	}	
}

namespace std {
	template<> 
	struct hash<ir::PTXOperand::DataType> {
	public:
		size_t operator()(const ir::PTXOperand::DataType& t) const {
			return (size_t)t;
		}
	};
}

#endif

