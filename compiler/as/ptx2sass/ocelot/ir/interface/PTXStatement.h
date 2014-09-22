/*! \file PTXStatement.h
	\date Monday January 19, 2009
	\author Andrew Kerr
	\brief The header file for the PTXStatement class
*/

#ifndef IR_PTXSTATEMENT_H_INCLUDED
#define IR_PTXSTATEMENT_H_INCLUDED

#include <vector>
#include <string>

#include <ocelot/ir/interface/PTXInstruction.h>

namespace ir {

	class PTXStatement {
	public:
		/*!	PTX directive types */
		enum Directive {
			Instr,			//! indicates this is an actual instruction
			AddressSize,
			CallTargets,
			Const,
			Entry,
			File,
			Func,
			FunctionPrototype,
			Global,
			Label,
			Local,
			Loc,
			Maxnreg,
			Maxntid,
			Maxnctapersm,
			Minnctapersm,
			Param,
			Pragma,
			Reg,
			Reqntid,
			Samplerref,
			Section,
			Shared,
			Sreg,
			Surfref,
			Target,
			Texref,
			Version,
			StartScope,		//! synthetic directive to indicate start of entry
			EndScope,		//! synthetic to indicate the end of an entry
			StartParam,		//! synthetic to indicate start of a parameter list
			EndParam,		//! synthetic to indicate end of a parameter list
			FunctionName,	//! synthetic to indicate the function name
			EndFuncDec, 	//! synthetic ending a function declaration
			Directive_invalid
		};

		union Data {
			PTXU8 u8;
			PTXU16 u16;
			PTXU32 u32;
			PTXU64 u64;

			PTXS8 s8;
			PTXS16 s16;
			PTXS32 s32;
			PTXS64 s64;

            PTXF32 f32;
            PTXF64 f64;

			PTXB8 b8;
			PTXB16 b16;
			PTXB32 b32;
			PTXB64 b64;
		};
		
		class Symbol
		{
		public:
			Symbol(const std::string& name = "", unsigned int o = 0);
		
		public:
			std::string name;
			unsigned int offset;
		};
		
		typedef std::vector< unsigned int > ArrayStrideVector;
		typedef std::vector< Data > ArrayVector;
		typedef std::vector< std::string > StringVector;
		typedef std::vector< PTXOperand::DataType > TypeVector;
		typedef std::vector< Symbol > SymbolVector;

		class StaticArray {
			public:
				ArrayStrideVector stride;
				ArrayVector values;
				PTXInstruction::Vec vec;
		
				SymbolVector symbols;
			
			public:
				std::string dimensions() const;
				std::string initializer( PTXOperand::DataType ) const;
		
				std::string valueAt( unsigned int index,
					PTXOperand::DataType t ) const;
		};
		
		/*!
			Attributes for statements
		*/
		enum Attribute {
			Visible,
			Extern,
			Weak,
			NoAttribute
		};
		
		enum TextureSpace {
			GlobalSpace,
			ParameterSpace,
			RegisterSpace,
			InvalidSpace
		};

	public:	
		static std::string toString( TextureSpace );
		static std::string toString( Attribute );
		static std::string toString( Data, PTXOperand::DataType );
		static std::string toString( Directive directive );
	
	public:
		/*! Indicates type of statement */
		Directive directive;

		PTXInstruction instruction;
	
		PTXOperand::DataType type;
	
		StaticArray array;
	
		std::string name;

		union {
			unsigned int sourceFile;
			int alignment;
			TextureSpace space;
			unsigned int addressSize;
		};
		
		std::string section_type;
		std::string section_name;

		union {
			unsigned int sourceLine;
			int major;
		};
		
		union {
			unsigned int sourceColumn;
			int minor;
		};
		
		unsigned int line;
		unsigned int column;
		
		Attribute attribute;

		StringVector targets;

		TypeVector returnTypes;
		TypeVector argumentTypes;
		
		bool isReturnArgument;
		
		PTXInstruction::AddressSpace ptrAddressSpace;

	public:
		PTXStatement( Directive directive = Directive_invalid );
		~PTXStatement();
		
		unsigned int bytes() const;
		unsigned int initializedBytes() const;
		unsigned int elements() const;
		unsigned int accessAlignment() const;

	public:
		/*! \brief Copy all of the initial data into a packed array */
		void copy(void* dest) const;
		std::string toString() const;
	};

}

#endif

