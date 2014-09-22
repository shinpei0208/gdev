/*!	\file   MoveEliminationPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday January 23, 2013
	\brief  The header file for the MoveEliminationPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief A transform to perform move elimination on a PTX kernel */
class MoveEliminationPass : public KernelPass
{
public:
	/*! \brief Create the pass, create dependencies */
	MoveEliminationPass();

public:
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
};


}

