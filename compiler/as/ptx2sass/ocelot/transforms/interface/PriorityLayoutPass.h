/*! \file   PriorityLayoutPass.h
	\author Gregory Diamso <gregory.diamos@gatech.edu>
	\date   Wednesday May 9, 2012
	\brief  The header file for the PriorityLayoutPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/transforms/interface/Pass.h>

// Standard Library Includes
#include <vector>

namespace transforms
{

/*! \brief Construct an instruction vector according to TF priorities */
class PriorityLayoutPass : public ImmutableKernelPass
{
public:
	typedef std::vector<ir::PTXInstruction> PTXInstructionVector;

public:
	/*! \brief The constructor sets the type */
	PriorityLayoutPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(const ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();
	
public:
	PTXInstructionVector instructions;
};

}

