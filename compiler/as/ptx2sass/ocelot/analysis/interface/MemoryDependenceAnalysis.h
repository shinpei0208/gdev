/*! \file   MemoryDependenceAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday June 29, 2013
	\file   The header file for the MemoryDependenceAnalysis class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>
#include <ocelot/analysis/interface/PTXInstructionDependenceGraph.h>

namespace analysis
{

/*! \brief Discover memory dependences in the program */
class MemoryDependenceAnalysis
: public KernelAnalysis, public PTXInstructionDependenceGraph
{
public:
	MemoryDependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);
};

}


