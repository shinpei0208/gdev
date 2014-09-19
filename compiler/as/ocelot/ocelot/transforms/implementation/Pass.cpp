/*! \file Pass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday September 15, 2009
	\brief The source file for the Pass class.
*/

#ifndef PASS_CPP_INCLUDED
#define PASS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/transforms/interface/Pass.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace transforms
{

Pass::Pass(Type t, const StringVector& a, const std::string& n)
	: type(t), analyses(a), name(n), _manager(0)
{

}

Pass::~Pass()
{

}

void Pass::setPassManager(PassManager* m)
{
	_manager = m;
}

analysis::Analysis* Pass::getAnalysis(const std::string& type)
{
	assert(_manager != 0);

	return _manager->getAnalysis(type);
}

const analysis::Analysis* Pass::getAnalysis(const std::string& type) const
{
	assert(_manager != 0);

	return _manager->getAnalysis(type);
}

void Pass::invalidateAnalysis(const std::string& type)
{
	assert(_manager != 0);

	return _manager->invalidateAnalysis(type);
}

void Pass::invalidateAllAnalyses()
{
	assert(_manager != 0);

	return _manager->invalidateAllAnalyses();
}

Pass::StringVector Pass::getDependentPasses() const
{
	return StringVector();
}

std::string Pass::toString() const
{
	return name;
}

ImmutablePass::ImmutablePass(const StringVector& a, const std::string& n) 
	: Pass(Pass::ImmutablePass, a, n)
{

}

ImmutablePass::~ImmutablePass()
{

}

ModulePass::ModulePass(const StringVector& a, const std::string& n) 
	: Pass( Pass::ModulePass, a, n)
{

}

ModulePass::~ModulePass()
{

}

KernelPass::KernelPass(const StringVector& a, const std::string& n)
	: Pass(Pass::KernelPass, a, n)
{

}

KernelPass::~KernelPass()
{

}

void KernelPass::initialize(const ir::Module& m)
{

}

void KernelPass::finalize()
{

}

ImmutableKernelPass::ImmutableKernelPass(const StringVector& a,
	const std::string& n)
	: Pass(Pass::ImmutableKernelPass, a, n)
{

}

ImmutableKernelPass::~ImmutableKernelPass()
{

}

BasicBlockPass::BasicBlockPass(const StringVector& a,
	const std::string& n)
	: Pass(Pass::BasicBlockPass, a, n)
{

}

BasicBlockPass::~BasicBlockPass()
{

}

}

#endif

