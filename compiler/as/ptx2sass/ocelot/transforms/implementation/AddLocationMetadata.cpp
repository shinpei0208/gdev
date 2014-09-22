/*! \file   AddLocationMetadataPass.cpp
	\date   Wednesday March 14, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the AddLocationMetadataPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/AddLocationMetadata.h>
#include <ocelot/ir/interface/PTXKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

AddLocationMetadataPass::AddLocationMetadataPass()
: KernelPass({}, "AddLocationMetadataPass")
{

}

void AddLocationMetadataPass::initialize(const ir::Module& m)
{
	// empty
}

void AddLocationMetadataPass::runOnKernel(ir::IRKernel& k)
{
	report("Adding location meta data to kernel " << k.name);

	for(auto block = k.cfg()->begin(); block != k.cfg()->end(); ++block)
	{
		if(block->instructions.empty()) continue;
		
		ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
			*block->instructions.front());
	
		ptx.metadata += " .location '" + k.getLocationString(ptx) + "'";
	
		report(" - " << ptx.toString() << " - " << ptx.metadata);
	}
}

void AddLocationMetadataPass::finalize()
{
	// empty
}

}


