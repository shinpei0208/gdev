/*!
	\file LLVMStatement.cpp
	\date Wednesday July 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the LLVMStatement class.
*/

#ifndef LLVM_STATEMENT_CPP_INCLUDED
#define LLVM_STATEMENT_CPP_INCLUDED

#include <ocelot/ir/interface/LLVMStatement.h>
#include <hydrazine/interface/debug.h>

namespace ir
{

	std::string LLVMStatement::toString( Linkage linkage )
	{
		switch( linkage )
		{
			case Private: return "private"; break;
			case LinkerPrivate: return "linker_private"; break;
			case Internal: return "internal"; break;
			case AvailableExternally: return "available_externally"; break;
			case LinkOnce: return "linkonce"; break;
			case External: return "external"; break;
			case Weak: return "weak"; break;
			case Common: return "common"; break;
			case Appending: return "appending"; break;
			case ExternWeak: return "extern_weak"; break;
			case LinkOnceOdr: return "linkonce_odr"; break;
			case WeakOdr: return "weak_odr"; break;
			case ExternallyVisible: return "externally visible"; break;
			case DllImport: return "dllimport"; break;
			case DllExport: return "dllexport"; break;
			case InvalidLinkage: break;
		}
		return "";	
	}
	
	std::string LLVMStatement::toString( Visibility visibility )
	{
		switch( visibility )
		{
			case Default: return "default"; break;
			case Hidden: return "hidden"; break;
			case Protected: return "protected"; break;
			case InvalidVisibility: break;
		}
		return "";
	}

	LLVMStatement::LLVMStatement( Type t, const LLVMInstruction* i ) 
		: instruction( 0 ), type( t ), linkage( InvalidLinkage ), 
		convention( LLVMInstruction::InvalidCallingConvention ), 
		visibility( InvalidVisibility ), 
		returnAttribute( LLVMInstruction::InvalidParameterAttribute ), 
		functionAttributes( 0 ), alignment( 1 ), space( 0 ), constant( false )
	{
		if( i != 0 )
		{
			instruction = static_cast< LLVMInstruction* >( i->clone() );
			assertM( type == Instruction, "Statement given non-zero " 
				<< "instruction pointer, but not specified as an " 
				<< "instruction statement." );
		}
		else
		{
		}
	}

	LLVMStatement::LLVMStatement( const LLVMStatement& s ) : instruction( 0 ), 
		type( s.type ), label( s.label ), linkage( s.linkage ),
		convention( s.convention ), visibility( s.visibility ), 
		operand( s.operand ), returnAttribute( s.returnAttribute ), 
		functionAttributes( s.functionAttributes ), section( s.section ), 
		alignment( s.alignment ), parameters( s.parameters ), space( s.space ), 
		constant( s.constant )
	{
		if( s.instruction != 0 )
		{
			instruction = static_cast< LLVMInstruction* >( 
				s.instruction->clone() );
			assertM( type == Instruction, "Statement given non-zero " 
				<< "instruction pointer, but not specified as an " );
		}
	}

	LLVMStatement::LLVMStatement( const LLVMInstruction& i ) 
		: type( Instruction ), linkage( InvalidLinkage ), 
		convention( LLVMInstruction::InvalidCallingConvention ), 
		visibility( InvalidVisibility ), 
		returnAttribute( LLVMInstruction::InvalidParameterAttribute ), 
		functionAttributes( 0 ), alignment( 1 ), space( 0 ), constant( false )
	{
		instruction = static_cast< LLVMInstruction* >( i.clone() );
	}

	LLVMStatement::LLVMStatement( const std::string& l ) 
		: instruction( 0 ), type( Label ), label( l )
	{
	}
	
	LLVMStatement::~LLVMStatement()
	{
		delete instruction;
	}
	
	const LLVMStatement& LLVMStatement::operator=( const LLVMStatement& s )
	{
		if( &s == this ) return *this;
		delete instruction;
		type = s.type;
		label = s.label;
		
		linkage = s.linkage;
		convention = s.convention;
		visibility = s.visibility;
		operand = s.operand;
		returnAttribute = s.returnAttribute;
		functionAttributes = s.functionAttributes;
		section = s.section;
		alignment = s.alignment;
		parameters = s.parameters;
		space = s.space;
		constant = s.constant;
		
		if( s.instruction != 0 )
		{
			instruction = static_cast< LLVMInstruction* >( 
				s.instruction->clone() );
			assertM( type == Instruction, "Statement given non-zero " 
				<< "instruction pointer, but not specified as an " 
				<< "instruction statement." );
		}
		else
		{
			instruction = 0;
		}
		return *this;
	}
	
	std::string LLVMStatement::toString() const
	{
		switch( type )
		{
			case Instruction:
			{
				return instruction->toString() + ";";
				break;
			}
			case Label:
			{
				return label + ":";
				break;
			}
			case FunctionDeclaration:
			case FunctionDefinition:
			{
				std::string result = (type == FunctionDefinition) 
					? "define " : "declare ";
				
				std::string link = toString( linkage );
				std::string visible = toString( visibility );
				std::string cc = LLVMInstruction::toString( convention );
				std::string retats = LLVMInstruction::toString( 
					returnAttribute );
				
				if( !link.empty() ) result += link + " ";
				if( !visible.empty() ) result += visible + " ";
				if( !cc.empty() ) result += cc + " ";
				if( !retats.empty() ) result += retats + " ";

				if( operand.valid() )
				{
					result += operand.type.toString() + " ";
				}
				else
				{
					result += "void ";
				}
				
				result += "@" + label + "( ";
				for( LLVMInstruction::ParameterVector::const_iterator 
					parameter = parameters.begin(); 
					parameter != parameters.end(); ++parameter )
				{
					if( parameter != parameters.begin() ) result += ", ";
					result += parameter->type.toString() 
						+ " " + parameter->toString();
				}
				result += " ) ";
				
				std::string atts = LLVMInstruction::functionAttributesToString( 
					functionAttributes );

				if( !atts.empty() ) result += atts + " ";
				if( !section.empty() ) result += section + " ";
			
				std::stringstream align;
				align << "align " << alignment;
				
				result += align.str() + ";";
				
				return result;
				break;
			}
			case TypeDeclaration:
			{
				return label + " = type " + operand.type.toString() + ";";
				break;
			}
			case VariableDeclaration:
			{
				std::string result = "@" + label + " = ";
				
				if( space != 0 )
				{
					std::stringstream address;
					address << "addrspace(" << space << ") ";
					result += address.str();
				}
				
				std::string link = toString( linkage );
				if( !link.empty() ) result += link + " ";
				
				if( constant )
				{
					result += "constant ";
				}
				else
				{
					result += "global ";
				}
				
				result += operand.type.toString();
				
				if( linkage != External )
				{
					result += " zeroinitializer";
				}
				
				std::stringstream align;
				align << ", align " << alignment;
			
				result += align.str();
				result += ";";

				return result;
				break;
			}
			case BeginFunctionBody:
			{
				return "{";
				break;
			}
			case EndFunctionBody:
			{
				return "}";
				break;
			}
			case NewLine:
			{
				break;
			}
			case InvalidType: break;
		}
		return "";
	}
}

#endif

