/*!
	\file Global.cpp
	
	\date Tuesday March 31, 2009
	\author Gregory Diamos
	
	\brief The source file for the Global class.	
*/

#ifndef GLOBAL_CPP_INCLUDED
#define GLOBAL_CPP_INCLUDED

#include <ocelot/ir/interface/Global.h>
#include <cassert>
#include <cstring>

namespace ir
{

	Global::Global() : local(false), pointer(0)
	{
		
	}
	
	Global::Global(char* p) : local(false), pointer(p)
	{
	
	}

	Global::Global(const ir::PTXStatement& s) : 
		local(!s.array.values.empty()), statement(s)
	{
		if(local) 
		{
			assert(statement.bytes() == statement.initializedBytes());
			unsigned int size = statement.initializedBytes();
			pointer = new char[size];
			
			unsigned int elementsize = PTXOperand::bytes(statement.type);
			unsigned int step = elementsize;
			char* end = (char*)pointer + size;
			PTXStatement::ArrayVector::iterator 
				ai = statement.array.values.begin();
			for(char* fi = (char*)pointer; fi < end; fi += step, ++ai )
			{
				assert(ai != statement.array.values.end());
				memcpy(fi, &ai->b8, elementsize);
			}
		}
		else
		{
			pointer = 0;
		}
	}

	Global::Global(const Global& g) : local(g.local)
	{
		statement = g.statement;
		if(local)
		{
			unsigned int size = statement.initializedBytes();
			pointer = new char[size];
			memcpy(pointer, g.pointer, size);
		}
		else
		{
			pointer = g.pointer;
		}
	}
	
	Global::~Global()
	{
		if( local )
		{
			delete[] (char*)pointer;
		}
	}
	
	Global& Global::operator=(const Global& g)
	{
		if(&g != this)
		{
			if(local)
			{
				delete[] (char*)pointer;
			}

			local = g.local;
			statement = g.statement;
			if(local)
			{
				unsigned int size = statement.initializedBytes();
				pointer = new char[size];
				memcpy(pointer, g.pointer, size);
			}
			else
			{
				pointer = g.pointer;
			}
			
		}
		return *this;
	}
	
	
	/*! \brief Get the address space of the global */
	PTXInstruction::AddressSpace Global::space() const {
		// todo fix this
		switch (statement.directive) {
			case ir::PTXStatement::Local:
				return PTXInstruction::Local;
			case ir::PTXStatement::Shared:
				return PTXInstruction::Shared;
			case ir::PTXStatement::Global:	// fall through
			default:
				break;			
		}
		return PTXInstruction::Global;
	}
	
	const std::string& Global::name() const
	{
		return statement.name;
	}
	
}

#endif

