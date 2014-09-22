/*! \brief  LoopUnrollingPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday May 2, 2012
	\brief  The header file for the LoopUnrollingPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief A class for splitting basic blocks larger than a specified size */
class LoopUnrollingPass : public KernelPass
{
public:
	/*! \brief The constructor sets pass type and dependencies */
	LoopUnrollingPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();

};

}


