/*! \file   DataDependenceAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday June 29, 2013
	\file   The header file for the DataDependenceAnalysis class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>
#include <ocelot/analysis/interface/PTXInstructionDependenceGraph.h>

namespace analysis
{

/*! \brief Discover all data dependences in a kernel */
class DataDependenceAnalysis
: public KernelAnalysis, public PTXInstructionDependenceGraph
{
public:
	DataDependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);

};

}


