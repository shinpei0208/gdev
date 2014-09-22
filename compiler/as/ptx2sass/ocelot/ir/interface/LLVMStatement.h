/*! \file LLVMStatement.h
	\date Wednesday July 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMStatement class.
*/

#ifndef LLVM_STATEMENT_H_INCLUDED
#define LLVM_STATEMENT_H_INCLUDED

#include <ocelot/ir/interface/LLVMInstruction.h>

namespace ir
{
	/*! \brief A class for LLVM declarations */
	class LLVMStatement
	{
		public:
			/*! \brief The type of statement */
			enum Type
			{
				Instruction,
				Label,
				FunctionDefinition,
				FunctionDeclaration,
				TypeDeclaration,
				VariableDeclaration,
				BeginFunctionBody,
				EndFunctionBody,
				NewLine,
				InvalidType
			};
			
			/*! \brief An llvm linkage type */
			enum Linkage
			{
				Private,
				LinkerPrivate,
				Internal,
				AvailableExternally,
				External,
				LinkOnce,
				Weak,
				Common,
				Appending,
				ExternWeak,
				LinkOnceOdr,
				WeakOdr,
				ExternallyVisible,
				DllImport,
				DllExport,
				InvalidLinkage
			};
			
			/*! \brief An llvm visibility type */
			enum Visibility
			{
				Default,
				Hidden,
				Protected,
				InvalidVisibility
			};
		
		public:
			static std::string toString( Linkage linkage );
			static std::string toString( Visibility visibility ); 
		
		public:
			/*! \brieg If this is an instruction, a pointer to the instruction 
				object.
				
				The pointer is owned by this class. 
			*/
			LLVMInstruction* instruction;
			
			/*! \brief The type of statement */
			Type type;
			
			/*! \brief The string if this is a label */
			std::string label;
			
			/*! \brief The linkage type if this is a declaration */
			Linkage linkage;
			
			/*! \brief The calling convention if this is a function */
			LLVMInstruction::CallingConvention convention;

			/*! \brief The visibility if this is a declaration */
			Visibility visibility;
			
			/*! \brief The operand if this is a variable declaration or the 
				returned type of a function call */
			LLVMInstruction::Operand operand;
			
			/*! \brief The return attribute if this is a function declaration */
			LLVMInstruction::ParameterAttribute returnAttribute;
			
			/*! \brief The attributes if this is a function delcaration */
			LLVMI32 functionAttributes;
			
			/*! \brief The section that this is delcared in */
			std::string section;
			
			/*! \brief The alignment if this is a declaration */
			LLVMI32 alignment;
			
			/*! \brief The set of parameter types if this is a function */
			LLVMInstruction::ParameterVector parameters;
			
			/*! \brief The address space if this is a variable */
			LLVMI32 space;

			/*! \brief Is this variable constant? */
			LLVMI1 constant;
			
		public:
			/*! \brief Sets out the instruction pointer and sets the type */
			LLVMStatement( Type type = InvalidType, 
				const LLVMInstruction* i = 0 );
			/*! \brief Construct a statement from an instruction */
			explicit LLVMStatement( const LLVMInstruction& i );
			/*! \brief Construction a statement from a label */
			explicit LLVMStatement( const std::string& l );
			/*! \brief Copy constructor for instruction */
			LLVMStatement( const LLVMStatement& s );
			/*! \brief Possibly cleans up the instruction pointer */
			~LLVMStatement();
			/*! \brief Assignment operator for instruction */
			const LLVMStatement& operator=( const LLVMStatement& s );
			/*! \brief Convert this statement into a string */
			std::string toString() const;
						
	};

}

#endif

