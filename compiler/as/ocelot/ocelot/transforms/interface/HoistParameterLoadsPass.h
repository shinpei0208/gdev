/*	\file   HoistParameterLoadsPass.h
	\date   December 14, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the HoistParameterLoadsPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>

// Forward Declarations
namespace ir { class PTXInstruction; }

namespace transforms
{

/*! \brief Hoists parameter loads to a dominating block

	Parameters are known to be constant for the lifetime of a kernel.  This
	pass exploits this fact to hoist the definition of parameters to create
	a single definition.  This pass reduces the number of parameter accesses
	at the cost of increased register pressure.
	
	It is also possible to apply this to memory base addresses for parameter
	spaces.  This effectively converts all parameter accesses into
	global accesss.

*/
class HoistParameterLoadsPass : public KernelPass
{
public:
	/*! \brief The default constructor */
	HoistParameterLoadsPass();

public:
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);

private:
	void _tryHoistingLoad(ir::ControlFlowGraph::iterator, ir::PTXInstruction*,
		ir::IRKernel&);

	ir::ControlFlowGraph::iterator _getTopLevelDominatingBlock(
		ir::IRKernel&, ir::ControlFlowGraph::iterator);

};

}

