/*! \file   DependenceAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday June 29, 2013
	\file   The header file for the DependenceAnalysis class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>
#include <ocelot/analysis/interface/PTXInstructionDependenceGraph.h>

namespace analysis
{

/*! \brief Discover all control and data dependences in a kernel */
class DependenceAnalysis
: public PTXInstructionDependenceGraph, public KernelAnalysis
{
public:
	DependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);

};

}


