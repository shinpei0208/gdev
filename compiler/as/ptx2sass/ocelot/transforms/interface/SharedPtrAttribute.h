/*!
	\file SharedPtrAttribute.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Sept 1, 2012
	\brief updates kernels by adding the shared memory base ptr
*/

#ifndef OCELOT_TRANSFORMS_SHAREDPTRATTRIBUTE_H_INCLUDED
#define OCELOT_TRANSFORMS_SHAREDPTRATTRIBUTE_H_INCLUDED

#include <ocelot/transforms/interface/Pass.h>

namespace transforms {

	/*! \brief A pass over a single kernel in a module */
	class SharedPtrAttribute : public ModulePass
	{
	public:
		
		SharedPtrAttribute(const std::string &n = "SharedPtrAttribute");
		virtual ~SharedPtrAttribute();
	
	public:
		/*! \brief Initialize the pass using a specific module */
		virtual void runOnModule(ir::Module& m);
	
		//! \brief returns true if any kernel contains .ptr.shared parameter attribute
		static bool testModule(const ir::Module &m);
		
	protected:
		std::string _getOrInsertExternShared(ir::Module &m);
		void _updateSharedPtrUses(ir::PTXKernel &k, std::string symbol);
		void _updateSharedUse(ir::PTXKernel &k, ir::Parameter &p, std::string symbol);

	};
}

#endif

