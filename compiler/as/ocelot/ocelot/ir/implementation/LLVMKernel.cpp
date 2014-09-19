/*! \file LLVMKernel.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Saturday August 1, 2009
	\brief The source file for the LLVMKernel class.
*/

#ifndef LLVM_KERNEL_CPP_INCLUDED
#define LLVM_KERNEL_CPP_INCLUDED

#include <ocelot/ir/interface/LLVMKernel.h>
#include <hydrazine/interface/Version.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/string.h>

#include <configure.h>

namespace ir
{
	LLVMKernel::LLVMKernel()
	{
		ISA = Instruction::LLVM;
	}

	LLVMKernel::LLVMKernel( const Kernel& k ) : Kernel( k )
	{
		ISA = Instruction::LLVM;
	}
	
	void LLVMKernel::push_back(const LLVMStatement& statement)
	{
		_statements.push_back(statement);
	}
	
	void LLVMKernel::push_front(const LLVMStatement& statement)
	{
		_statements.push_front(statement);
	}
	
	void LLVMKernel::assemble()
	{
		_code.clear();
		
		hydrazine::Version version;
		
		_code += "; Code assembled by Ocelot LLVMKernel " + version.toString() 
			+ "\n\n";
		
		for( LLVMStatementVector::const_iterator 
			statement = llvmStatements().begin(); 
			statement != llvmStatements().end(); ++statement )
		{
			if( statement->type == LLVMStatement::Instruction ) _code += "\t";
			_code += statement->toString() + "\n";
		}
	}
	
	bool LLVMKernel::assembled() const
	{
		return !_code.empty();
	}
	
	const std::string& LLVMKernel::code() const
	{
		return _code;
	}
	
	std::string LLVMKernel::numberedCode() const
	{
		return hydrazine::addLineNumbers( _code );
	}
	
	const LLVMKernel::LLVMStatementVector& LLVMKernel::llvmStatements() const
	{
		return _statements;
	}
}

#endif

