/*! \file   CycleAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday May 31, 2013
	\brief  The source file for the CycleAnalysis class.
*/

// Ocelot Incudes
#include <ocelot/analysis/interface/CycleAnalysis.h>

#include <ocelot/ir/interface/IRKernel.h>

// Standard Library Includes
#include <stack>

namespace analysis
{

CycleAnalysis::CycleAnalysis()
: KernelAnalysis("CycleAnalysis")
{

}

void CycleAnalysis::analyze(ir::IRKernel& kernel)
{
	typedef std::stack<iterator> BlockStack;
	typedef std::unordered_set<iterator> BlockSet;

	BlockSet   visited;
	BlockStack stack;
	
	// DFS, edges that point to visited nodes are back edges
	// NOTE: assumes no duplicate edges
	stack.push(kernel.cfg()->begin());
	
	while(!stack.empty())
	{
		auto block = stack.top();
		stack.pop();
		
		for(auto edge = block->out_edges.begin();
			edge != block->out_edges.end(); ++edge)
		{
			if(visited.insert((*edge)->tail).second)
			{
				stack.push((*edge)->tail);
			}
			else
			{
				_backEdges.insert(*edge);
			}
		}
	}
}

CycleAnalysis::EdgeList CycleAnalysis::getAllBackEdges() const
{
	return EdgeList(_backEdges.begin(), _backEdges.end());
}

bool CycleAnalysis::isBackEdge(edge_iterator edge) const
{
	return _backEdges.count(edge) != 0;
}

}

