/*! \file ILStatement.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 27, 2010
 *  \brief The implementation file for the IL Statement class.
 */

// Ocelot includes
#include <ocelot/ir/interface/ILStatement.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace ir
{
	ILStatement::ILStatement(Type type) : instruction(0), type(type)
	{
	}

	ILStatement::ILStatement(const ILInstruction &i) 
		: instruction(0), type(Instruction)
	{
		instruction = static_cast<ILInstruction*>(i.clone());
	}

	ILStatement::ILStatement(const ILStatement& s) : instruction(0), 
		type(s.type), operands(s.operands), arguments(s.arguments)
	{
		if( s.instruction != 0 )
		{
			instruction = static_cast< ILInstruction* >( 
				s.instruction->clone() );
			assertM( type == Instruction, "Statement given non-zero " 
				<< "instruction pointer, but not specified as an "
				<< "instruction statement." );
		}
	}

	ILStatement::~ILStatement()
	{
		delete instruction;
	}
	
	std::string ILStatement::toString() const
	{
		switch(type)
		{
			case Instruction: return instruction->toString();
			case ConstantBufferDcl:
			{
				std::stringstream stream;

				stream << "dcl_cb ";
				stream << operands[0].srcString();

				return stream.str();
			}
			case LiteralDcl:
			{
				std::stringstream stream;

				stream << "dcl_literal ";
				stream << operands[0].dstString() << ", ";
				stream << std::hex;
				stream << "0x" << arguments[0] << ", ";
				stream << "0x" << arguments[1] << ", ";
				stream << "0x" << arguments[2] << ", ";
				stream << "0x" << arguments[3];

				return stream.str();
			}
			case LocalDataShareDcl:
			{
				std::stringstream s;

				s << "dcl_lds_id(1) ";
				s << arguments[0];

				return s.str();
			}
			case OtherDeclarations: return \
							  "il_cs_2_0\n"
							  "dcl_max_thread_per_group 512\n"
							  "dcl_raw_uav_id(0)\n"
							  "dcl_arena_uav_id(8)";
			default:
			{
				assertM(false, "Statement type "
						<< type
						<< " not supported");
				break;
			}
		}
		
		return "";
	}
}

