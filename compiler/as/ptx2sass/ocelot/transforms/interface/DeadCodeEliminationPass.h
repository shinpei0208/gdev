/*! \file   DeadCodeEliminationPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday July 21, 2011
	\brief  The header file for the DeadCodeEliminationPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief A transform to perform dead code elimination on a PTX kernel */
class DeadCodeEliminationPass : public KernelPass
{
public:
	/*! \brief Create the pass, create dependencies */
	DeadCodeEliminationPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();
};


}

