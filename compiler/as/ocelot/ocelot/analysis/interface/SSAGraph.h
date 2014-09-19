/*! \file SSAGraph.h
	\date Saturday June 27, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the SSAGraph class.	
*/

#pragma once

// Ocelot Includes
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <map>

namespace analysis
{

/*! \brief SSA Graph, used as a helper by the dataflow graph to convert into
	SSA form. */
class SSAGraph
{
private:
	typedef std::unordered_map<DataflowGraph::Register, 
		DataflowGraph::Register> RegisterMap;

	typedef std::map<DataflowGraph::RegisterId,
		DataflowGraph::Register> RegisterIdMap;

	class Block
	{
	public:
		RegisterMap regs;
		RegisterMap aliveInMap;
	};

	typedef std::unordered_map<DataflowGraph::iterator, Block> BlockMap;

private:
	DataflowGraph& _graph;
	BlockMap _blocks;
	DataflowGraph::SsaType _form;

	void _initialize(Block& b, DataflowGraph::iterator it, 
		DataflowGraph::RegisterId& current);		
	void _insertPhis();
	void _updateIn();
	void _updateOut();
	void _minimize();
	void _gssa(DataflowGraph::RegisterId&);
	bool _isPossibleDivBranch(
		const DataflowGraph::InstructionVector::iterator &) const;

public:
	SSAGraph( DataflowGraph& graph,
		DataflowGraph::SsaType form = DataflowGraph::SsaType::Default);
	void toSsa();
	void fromSsa();
};

}


