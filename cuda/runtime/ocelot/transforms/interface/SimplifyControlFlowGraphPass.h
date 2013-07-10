/*	\file   SimplifyControlFlowGraphPass.h
	\date   Tuesday January 31, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the SimplifyControlFlowGraphPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief Convert CFG structures into a canonical form */
class SimplifyControlFlowGraphPass : public KernelPass
{
public:
	/*! \brief The default constructor */
	SimplifyControlFlowGraphPass();

public:
	/*! \brief Should this pass merge exit blocks? */
	bool mergeExitBlocks;
	/*! \brief Should this pass delete empty blocks? */
	bool deleteEmptyBlocks;

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);

	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		

	/*! \brief Finalize the pass */
	void finalize();

private:
	/*! \brief Merge all exit points into a single basic block
		(ret for a function, exit for a global) 
	
		\return True if this pass changes the CFG.
	*/
	bool _mergeExitBlocks(ir::IRKernel& k);


	/*! \brief Delete empty blocks.
	
		\return True if this pass changes the CFG.
	*/
	bool _deleteEmptyBlocks(ir::IRKernel& k);

	/*! \brief Delete unconnected blocks.
	
		\return True if this pass changes the CFG.
	*/
	bool _deleteUnconnectedBlocks(ir::IRKernel& k);

	/*! \brief Merge a basic block into its predecessor if possible
	
		\return True if this pass changes the CFG.
	*/
	bool _mergeBlockIntoPredecessor(ir::IRKernel& k);

	/*! \brief Simplify the terminator instruction on basic blocks.
	
		\return True if this pass changes the CFG.
	*/
	bool _simplifyTerminator(ir::IRKernel& k);

};

}


