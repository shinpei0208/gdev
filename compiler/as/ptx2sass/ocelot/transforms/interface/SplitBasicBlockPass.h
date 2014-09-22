/*! \brief  SplitBasicBlockPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday August 12, 2011
	\brief  The header file for the SplitBasicBlockPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief A class for splitting basic blocks larger than a specified size */
class SplitBasicBlockPass : public KernelPass
{
public:
	/*! \brief The constructor sets the block size */
	SplitBasicBlockPass(unsigned int maxSize = 50);

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();

public:
	void setMaximumBlockSize(unsigned int s);

private:
	unsigned int _maxSize;

};

}


