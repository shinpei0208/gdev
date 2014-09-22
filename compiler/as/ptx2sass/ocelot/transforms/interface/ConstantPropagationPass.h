/*!	\file   ConstantPropagationPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday November 9, 2012
	\brief  The header file for the ConstantPropagationPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief A transform to perform constant propagation on a PTX kernel */
class ConstantPropagationPass : public KernelPass
{
public:
	/*! \brief Create the pass, create dependencies */
	ConstantPropagationPass();

public:
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
};


}

