/*! \file   HammockGraphAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday May 31, 2013
	\brief  The header file for the HammockGraphAnalysis class.
*/

#pragma once

// Ocelot Incudes
#include <ocelot/analysis/interface/Analysis.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>

// Standard Library Includes
#include <list>
#include <unordered_set>

namespace analysis
{

/*! \brief Analysis that discovers hammock graphs in a kernel. */
class HammockGraphAnalysis: public KernelAnalysis
{
public:
	typedef ir::ControlFlowGraph CFG;
	
	typedef CFG::const_iterator const_iterator;
	typedef CFG::iterator       iterator;

public:
	class Hammock
	{
	public:
		typedef std::list<Hammock> HammockList;
	
	public:
		Hammock(Hammock* parent = 0, iterator en = iterator(),
			iterator ex = iterator());
	
	public:
		bool isLeaf() const;
	
	public:
		Hammock*    parent;
		HammockList children;
		iterator    entry; // If the entry and exit are the same, the hammock
		iterator    exit;  //  contains a single block.
	};
	
	typedef Hammock::HammockList  HammockList;
	typedef HammockList::iterator hammock_iterator;
	
public:
	/*! \brief Create the analysis */
	HammockGraphAnalysis();

	/*! \brief discovers all hammocks in the program */
	void analyze(ir::IRKernel& kernel);

public:
	/*! \brief Get the inner-most hammock of the specified block */
	const Hammock* getHammock(const_iterator block) const;

	/*! \brief Gets the root of the hammock graph */
	const Hammock* getRoot() const;

private:
	typedef std::unordered_map<const_iterator, Hammock*> HammockMap;
	
private:
	Hammock    _root;
	HammockMap _blockToHammockMap;

};

}

