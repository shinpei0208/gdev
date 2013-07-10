/*! \file Kernel.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief implements the Kernel base class
*/

#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

#include <deque>
#include <map>

#include <ocelot/ir/interface/Local.h>
#include <ocelot/ir/interface/Parameter.h>
#include <ocelot/ir/interface/Instruction.h>

namespace ir {
	class Module;
}

namespace ir {
	/*! Base class for kernels */
	class Kernel {
	public:
		/*!	\brief Vector of parameters */
		typedef std::vector<Parameter> ParameterVector;
		/*! \brief Map from names to parameters */
		typedef std::map<std::string, Parameter> ParameterMap;
		/*! \brief Map from names to local variables */
		typedef std::map<std::string, Local> LocalMap;
		/*! \brief A type of an Id */
		typedef unsigned int Id;

	protected:
		/*! \brief Is this kernel a function? */
		bool _function;
		
	public:
		/*!	Constructs an empty kernel */
		Kernel(Instruction::Architecture isa = Instruction::Unknown,
			const std::string& name = "", bool isFunction = false,
			const ir::Module* module = 0, Id id = 0);
		/*!	Destructs kernel */
		virtual ~Kernel();
		/*! \brief Write this kernel to a parseable string */
		virtual void write(std::ostream& stream) const;

	public:
		/*!	Returns a pointer to a parameter identified by 'name' */		
		Parameter* getParameter(const std::string& name);
		/*!	Returns a const pointer to a parameter identified by 'name' */
		const Parameter* getParameter(const std::string& name) const;
		
		/*! Inserts a parameter into the appropriate data structure
			\param param defines parameter to insert
			\param asParameter if true, assumes this is param declaration that
				is not a kernel argument
		*/
		void insertParameter(const ir::Parameter &param, bool asParameter=true);

	public:	
		/*! \brief Is this kernel actually a function, not a kernel? */
		bool function() const;
		
	public:
		/*!	[mangled] name of kernel within module */
		std::string name;
		/*!	Unique id of the kernel within the module */
		Id id;
		/*!	Instruction Set Architecture of the kernel */
		Instruction::Architecture ISA;
		/*!	Set of parameters in order specified in the kernel's definition */
		ParameterVector arguments;
		/*! Set of parameters that are not kernel arguments */
		ParameterMap parameters;
		/*! \brief Local variables */
		LocalMap locals;
		/*!	Pointer to the module this kernel belongs to */
		const Module* module;
	};

}

std::ostream& operator<<(std::ostream& s, const ir::Kernel& k);

#endif

