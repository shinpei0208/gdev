/*! \file RemoveBarrierPass.h
	\date Tuesday September 15, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the RemoveBarrierPass class
*/

#ifndef REMOVE_BARRIER_PASS_H_INCLUDED
#define REMOVE_BARRIER_PASS_H_INCLUDED

#include <ocelot/transforms/interface/Pass.h>
#include <ocelot/analysis/interface/DataflowGraph.h>

namespace ir
{
	class PTXKernel;
	class ExternalFunctionSet;
}

namespace transforms
{
/*! \brief A class for a pass that removes all barriers from a PTX kernel

	This implementation leaves identifies barriers and
	splits the basic block containing them into two.  The first block
	contains all of the code before the barrier, spill instructions to
	a stack in local memory, and a tail call to resume this kernel.
	A local variable is allocated on the stack to indicate the program
	entry point.
	
	The program entry point is augmented to include a conditinal branch to
	the second block of each split barrier depending on the program entry 
	point variable.  The second block is augmented with code to load the 
	live variables from the local memory stack.	
*/
class RemoveBarrierPass : public KernelPass
{
private:
	ir::PTXKernel*                 _kernel;
	unsigned int                   _reentryPoint;
	unsigned int                   _kernelId;
	unsigned int                   _spillBytes;
	const ir::ExternalFunctionSet* _externals;
	
private:
	analysis::DataflowGraph& _dfg();
	analysis::DataflowGraph::RegisterId _tempRegister( );
	void _addSpillCode( analysis::DataflowGraph::iterator block, 
		analysis::DataflowGraph::iterator target, 
		const analysis::DataflowGraph::RegisterSet& alive,
		bool isBarrier );
	void _addRestoreCode( analysis::DataflowGraph::iterator block, 
		const analysis::DataflowGraph::RegisterSet& alive );
	void _addEntryPoint( analysis::DataflowGraph::iterator block );
	void _removeBarrier( analysis::DataflowGraph::iterator block, 
		unsigned int instruction );
	void _addLocalVariables();
	void _runOnBlock( analysis::DataflowGraph::iterator block );

public:
	RemoveBarrierPass( unsigned int kernelId = 0,
		const ir::ExternalFunctionSet* externals = 0 );
	
public:
	void initialize( const ir::Module& m );
	void runOnKernel( ir::IRKernel& k );		
	void finalize( );
	
public:
	bool usesBarriers;

};

}

#endif

