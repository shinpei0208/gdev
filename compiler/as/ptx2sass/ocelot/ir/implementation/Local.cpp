/*!
	\file Local.cpp
	\author Gregory Diamos <gregory.diamos@gatech>
	\date Thursday September 17, 2009
	\brief The source file for the Local class
*/

#ifndef LOCAL_CPP_INCLUDED
#define LOCAL_CPP_INCLUDED

#include <ocelot/ir/interface/Local.h>

#include <hydrazine/interface/debug.h>

namespace ir
{
	Local::Local( const PTXStatement& statement ): type( statement.type ), 
		name( statement.name ), alignment( statement.alignment ), 
		vector( statement.array.vec ), elements( statement.elements() ), 
		attribute( statement.attribute ), offset( 0 )
	{
		switch( statement.directive )
		{
			case PTXStatement::Local: space = PTXInstruction::Local; break;
			case PTXStatement::Shared: space = PTXInstruction::Shared; break;
			case PTXStatement::Param: space = PTXInstruction::Param; break;
			default:
			{
				assertM( space == PTXInstruction::Local 
					|| space == PTXInstruction::Shared 
					|| space == PTXInstruction::Param, 
					"Directive is not valid for local variables." );
			}
		}	
	}
	
	Local::Local() : space( PTXInstruction::AddressSpace_Invalid ), 
		type( PTXOperand::TypeSpecifier_invalid ), alignment( 1 ), 
		vector( PTXOperand::v1 ), elements( 1 ), 
		attribute( PTXStatement::NoAttribute ), offset( 0 )
		
	{
	
	}
	
	unsigned int Local::getSize() const
	{
		return getElementSize() * elements;
	}

	unsigned int Local::getElementSize() const
	{
		switch (type) 
		{
			case PTXOperand::pred: /* fall through */
			case PTXOperand::b8: /* fall through */
			case PTXOperand::u8: /* fall through */
			case PTXOperand::s8: /* fall through */
				return sizeof(PTXB8) * vector;
			case PTXOperand::u16: /* fall through */
			case PTXOperand::s16: /* fall through */
			case PTXOperand::b16: /* fall through */
				return sizeof(PTXU16) * vector;
			case PTXOperand::u32: /* fall through */
			case PTXOperand::s32: /* fall through */
			case PTXOperand::b32: /* fall through */
			case PTXOperand::f32: /* fall through */
				return sizeof(PTXU32) * vector;
			case PTXOperand::u64: /* fall through */
			case PTXOperand::s64: /* fall through */
			case PTXOperand::b64: /* fall through */
			case PTXOperand::f64: /* fall through */
				return sizeof(PTXU64) * vector;
			default: break;
		}
		return 0;
	}

	unsigned int Local::getAlignment() const
	{
		return std::max( getElementSize(), alignment );
	}
			
	PTXStatement Local::statement() const
	{
		PTXStatement statement;
		
		if( space == PTXInstruction::Shared )
		{
			statement.directive = PTXStatement::Shared;
		}
		else
		{		
			statement.directive = PTXStatement::Local;
		}
		
		statement.name = name;
		statement.alignment = alignment;
		statement.type = type;
		statement.array.vec = vector;
		statement.array.stride.push_back( elements );
		statement.attribute = attribute;
		
		return statement;
	}

	std::string Local::toString() const
	{
		return statement().toString();
	}
}

#endif

