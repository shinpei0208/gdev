/*! \file   ExternalFunctionSet.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday April 6, 2011
	\brief  The header file for the ExternalFunctionSet class.
*/

#ifndef EXTERNAL_FUNCTION_SET_H_INCLUDED
#define EXTERNAL_FUNCTION_SET_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/PTXKernel.h>

// Forward Declarations
namespace llvm { class Module; }

namespace ir
{

/*! \brief Holds a collection of external functions

	The idea here is to allow arbitrary PTX functions to call into external
	functions.  The major issue is that we need a portable way of passing
	parameters for different calling conventions.  The selected way of doing
	this is using LLVM to jit-compile a layer for performing the call.  This is
	done on both the emulator and the LLVM jit.  For other targets, we either
	use host RPCs or rely on the lower layer to provide an implementation of
	built-in functions.
 */
class ExternalFunctionSet
{
public:
	class ExternalFunction
	{
	public:
		ExternalFunction(const std::string& identifier = "",
			void* functionPointer = 0, llvm::Module* m = 0);
	
	public:
		void call(void* parameters, const ir::PTXKernel::Prototype& p);
		const std::string& name() const;
		void* functionPointer() const;
		std::string mangledName() const;
		
	private:
		typedef void (*ExternalCallType)(void*);
		
	private:
		std::string      _name;
		void*            _functionPointer;
		llvm::Module*    _module;
		ExternalCallType _externalFunctionPointer;
	};

	typedef std::map<std::string, ExternalFunction> FunctionSet;

public:
	/*! \brief Establish a link with llvm */
	ExternalFunctionSet();
	
	/*! \brier Teardown the link with llvm */
	~ExternalFunctionSet();

public:
	/*! \brief Add a new external function that is callable from PTX kernels */
	void add(const std::string& name, void* pointer);
	
	/*! \brief Remove a function by name */
	void remove(const std::string& name);
	
	/*! \brief Get a callable external function or 0 if it doesn't exist */
	ExternalFunction* find(const std::string& name) const;
	
private:
	FunctionSet   _functions;
	llvm::Module* _module;

};

}

#endif

