/*! \file   EnforceLockStepExecutionPass.h
	\date   Wednesday April 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the  EnforceLockStepExecutionPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/Instruction.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

// Standard Library Includes
#include <unordered_map>
#include <unordered_set>

namespace transforms
{

/*! \brief Insert instructions to modify the active mask to enforce lock-step
           execution down a divergent subgraph */
class EnforceLockStepExecutionPass : public ::transforms::KernelPass
{
public:
	/*! \brief The default constructor sets the type */
	EnforceLockStepExecutionPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);
	/*! \brief Finalize the pass */
	void finalize();

};

}


