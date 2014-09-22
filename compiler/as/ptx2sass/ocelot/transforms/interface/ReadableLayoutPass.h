/*! \file   ReadableLayoutPass.h
	\author Gregory Diamos <diamos@nvidia.com>
	\date   Wednesday July 11, 2012
	\brief  The header file for the ReadableLayoutPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>


namespace transforms
{

/*! \brief Construct a basic block vector of instructions */
class ReadableLayoutPass : public KernelPass
{
public:
	typedef ir::ControlFlowGraph::BlockPointerVector BlockPointerVector;
	typedef ir::ControlFlowGraph::iterator           iterator;

public:
	/*! \brief The constructor sets the type */
	ReadableLayoutPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();
	
public:
	BlockPointerVector blocks;

private:
	bool _isCyclicDependency(iterator predecessor, iterator successor);
};

}


