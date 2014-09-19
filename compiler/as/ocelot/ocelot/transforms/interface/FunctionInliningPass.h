/*! \file   FunctionInliningPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday June 20, 2012
	\brief  The header file for the FunctionInliningPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>

namespace transforms
{

/*! \brief A transform to perform function inlining within a PTX kernel */
class FunctionInliningPass : public KernelPass
{
public:
	/*! \brief Create the pass, create dependencies
	
		\param threshold The maximum size of a function (in instructions)
			to inline
	 */
	FunctionInliningPass(unsigned int threshold = 1000);

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();

public:
	StringVector getDependentPasses() const;
	
public:
	unsigned int thresholdToInline;

private:
	void _getFunctionsToInline(ir::IRKernel& k);
	void _inlineSelectedFunctions(ir::IRKernel& k);

private:
	class FunctionCallDescriptor
	{
	public:
		FunctionCallDescriptor(ir::ControlFlowGraph::instruction_iterator c,
			ir::ControlFlowGraph::iterator b, const ir::IRKernel* k,
			bool linked = false);
	
	public:
		ir::ControlFlowGraph::instruction_iterator call;
		ir::ControlFlowGraph::iterator             basicBlock;
		const ir::IRKernel*                        calledKernel;
	
	public:
		bool linked;
	};

	typedef std::vector<FunctionCallDescriptor> FunctionDescriptorVector;

private:
	void _updateLinks(ir::ControlFlowGraph::iterator splitBlock,
		FunctionDescriptorVector::iterator call);

private:
	FunctionDescriptorVector _calls;
	unsigned int             _nextRegister;
};

}

