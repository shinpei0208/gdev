/*! \file   ConvergentRegionAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday May 4, 2012
	\brief  The source file for the ConvergentRegionAnalysis class.
*/

// Ocelot Incudes
#include <ocelot/analysis/interface/ConvergentRegionAnalysis.h>

#include <ocelot/ir/interface/IRKernel.h>

// Standard Library Includes
#include <cassert>

namespace analysis
{

ConvergentRegionAnalysis::ConvergentRegionAnalysis()
: KernelAnalysis("ConvergentRegionAnalysis",
	{"DivergenceAnalysis", "PostDominatorTreeAnalysis"})
{

}

void ConvergentRegionAnalysis::analyze(ir::IRKernel& kernel)
{
	_regions.clear();
	
	Region region = 0;
	for(auto block = kernel.cfg()->begin();
		block != kernel.cfg()->end(); ++block )
	{
		_regions.insert(std::make_pair(block, region++));
	}
	
	/* TODO, create larger regions
	bool changed = true;
	
	while(changed)
	{
		for(auto block = kernel.cfg()->begin();
			block != kernel.cfg()->end(); ++block )
		{
			if()
		}
	}
	*/
}

ConvergentRegionAnalysis::Region ConvergentRegionAnalysis::getRegion(
	const_iterator block) const
{
	auto region = _regions.find(block);
	assert(region != _regions.end());
	
	return region->second;
}
	

}


