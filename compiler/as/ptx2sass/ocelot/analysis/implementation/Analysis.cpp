/*! \file   Analysis.cpp
	\author Gregory Diamos <gdiamos@gatech.edu>
	\date   Saturday May 7, 2011
	\brief  The source file for the Analysis class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>

#include <ocelot/transforms/interface/PassManager.h>

// Standard Library Includes
#include <cassert>

namespace analysis
{

Analysis::Analysis(const std::string& n, const StringVector& r)
: name(n), required(r), _manager(0)
{

}

Analysis::~Analysis()
{

}

void Analysis::configure(const StringVector& options)
{
	
}

KernelAnalysis::KernelAnalysis(const std::string& n,
	const StringVector& r)
: Analysis(n, r)
{

}

ModuleAnalysis::ModuleAnalysis(const std::string& n,
	const StringVector& r)
: Analysis(n, r)
{

}

void Analysis::setPassManager(transforms::PassManager* m)
{
	_manager = m;
}

Analysis* Analysis::getAnalysis(const std::string& name)
{
	assert(_manager != 0);
	return _manager->getAnalysis(name);
}

const Analysis* Analysis::getAnalysis(const std::string& name) const
{
	assert(_manager != 0);
	return _manager->getAnalysis(name);
}

void Analysis::invalidateAnalysis(const std::string& name)
{
	assert(_manager != 0);
	_manager->invalidateAnalysis(name);
}

}

