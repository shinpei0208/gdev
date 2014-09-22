/*! \file   SafeRegionAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday May 4, 2012
	\brief  The header file for the SafeRegionAnalysis class.
*/

#pragma once

// Ocelot Incudes
#include <ocelot/analysis/interface/Analysis.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>

// Standard Library Includes
#include <list>
#include <unordered_map>

namespace analysis
{

/*! \brief A class for tracking program regions that can be safely vectorized */
class SafeRegionAnalysis: public KernelAnalysis
{
public:
	typedef ir::ControlFlowGraph CFG;

	typedef CFG::const_iterator const_iterator;
	typedef CFG::iterator       iterator;

	class SafeRegion
	{
	public:
		typedef std::list<SafeRegion> SafeRegionList;
	
	public:
		SafeRegion(SafeRegion* parent = nullptr);
	
	public:
		bool isLeaf() const;
	
	public:
		SafeRegion*    parent;
		SafeRegionList children;
		iterator       block; // If this is a leaf node, the block
		
	public:
		bool doesNotDependOnSideEffects;
	};
	
	typedef std::unordered_map<const_iterator, SafeRegion*> SafeRegionMap;

public:
	/*! \brief Create the analysis */
	SafeRegionAnalysis();

	/*! \brief Computes an up to date set of regions */
	void analyze(ir::IRKernel& kernel);

public:
	/*! \brief Get the region of the specified block */
	const SafeRegion* getRegion(const_iterator block) const;
	/*! \brief Get the root of the program */
	const SafeRegion* getRoot() const;

private:
	SafeRegion    _root;
	SafeRegionMap _regions;
	
};

}

