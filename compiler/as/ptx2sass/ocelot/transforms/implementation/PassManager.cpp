/*! \file PassManager.cpp
	\date Thursday September 16, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the PassManager class
*/


// Ocelot Includes
#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/transforms/interface/PassFactory.h>
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/analysis/interface/Analysis.h>
#include <ocelot/analysis/interface/AnalysisFactory.h>

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

typedef PassManager::AnalysisMap  AnalysisMap;
typedef PassManager::IRKernel     IRKernel;
typedef PassManager::Module       Module;
typedef PassManager::PassWaveList PassWaveList;

typedef std::unordered_map<std::string, unsigned int> PassUseCountMap;

static PassUseCountMap getPassUseCounts(const PassWaveList& waves)
{
	PassUseCountMap uses;
	
	for(auto wave : waves)
	{
		for(auto pass : wave)
		{
			for(auto analysisType : pass->analyses)
			{
				auto use = uses.find(analysisType);
			
				if(use == uses.end())
				{
					use = uses.insert(std::make_pair(analysisType, 1)).first;
				}
				else
				{
					use->second += 1;
				}
				
				report(" Recording future use of analysis " << analysisType
					<< " (" << use->second << " total) by " << pass->name);
			}
		}
	}

	return uses;
}

static PassUseCountMap getPassUseCounts(const PassWaveList& waves,
	const ir::Module& module)
{
	PassUseCountMap uses;
	
	for(auto wave : waves)
	{
		for(auto pass : wave)
		{
			for(auto analysisType : pass->analyses)
			{
				auto use = uses.find(analysisType);
			
				unsigned int useCount = 0;
				
				switch(pass->type)
				{
				case Pass::ImmutablePass: // fall through
				case Pass::ModulePass:
				{
					useCount = 1;
				}
				break;
				case Pass::ImmutableKernelPass: // fall through
				case Pass::KernelPass:
				{
					useCount = module.kernels().size();
				}
				break;
				{
					useCount = module.kernels().size();
				}
				break;
				case Pass::BasicBlockPass:
				{
					for(auto function = module.kernels().begin();
						function != module.kernels().end(); ++function)
					{
						useCount += function->second->cfg()->size();
					}
				}
				break;
				default : break;
				}
				
				if(useCount == 0) continue;
				
				if(use == uses.end())
				{
					use = uses.insert(std::make_pair(
						analysisType, useCount)).first;
				}
				else
				{
					use->second += useCount;
				}
				
				report(" Recording future use of analysis " << analysisType
					<< " (" << use->second << " total) by " << pass->name);
			}
		}
	}

	return uses;
}
	
static void freeUnusedDataStructures(PassUseCountMap& uses,
	AnalysisMap& analyses)
{
	Pass::StringVector freedAnalyses;
	
	for(auto& use : uses)
	{
		if(use.second == 0)
		{
			freedAnalyses.push_back(use.first);
		}
	}
	
	for(auto& analysisType : freedAnalyses)
	{
		auto use = uses.find(analysisType);
		
		assertM(use != uses.end(), "Could not find entry for "
			<< analysisType);
		
		if(use->second == 0)
		{
			report("  Freeing analysis " << analysisType);
			uses.erase(use);
			
			auto analysis = analyses.find(analysisType);
			
			if(analysis != analyses.end())
			{
				delete analysis->second;

				analyses.erase(analysis);
			}
		}
	}
}

static void allocateDependencies(PassUseCountMap& uses,
	analysis::Analysis* newAnalysis,
	AnalysisMap& analyses, IRKernel* function, PassManager* manager);
static void freeDependencies(PassUseCountMap& uses,
	analysis::Analysis* newAnalysis);

static void allocateDataStructure(PassUseCountMap& uses,
	const std::string& analysisType,
	AnalysisMap& analyses, IRKernel* function, PassManager* manager)
{
	if(analyses.count(analysisType) != 0) return;

	report("  Creating analysis " << analysisType);
	
	auto newAnalysis = analysis::AnalysisFactory::createAnalysis(analysisType);
	assert(newAnalysis != nullptr);

	newAnalysis->setPassManager(manager);

	// allocate dependencies (use-counts)
	allocateDependencies(uses, newAnalysis, analyses, function, manager);
	
	auto functionAnalysis = static_cast<analysis::KernelAnalysis*>(
		newAnalysis);

	functionAnalysis->analyze(*function);

	// free dependencies (use-counts)
	freeDependencies(uses, newAnalysis);
	
	analyses.insert(std::make_pair(newAnalysis->name, newAnalysis));
}

static void allocateDependencies(PassUseCountMap& uses,
	analysis::Analysis* newAnalysis,
	AnalysisMap& analyses, IRKernel* function, PassManager* manager)
{
	// increment use count
	for(auto& type : newAnalysis->required)
	{
		auto use = uses.find(type);
			
		if(use == uses.end())
		{
			use = uses.insert(std::make_pair(type, 1)).first;
		}
		else
		{
			++use->second;
		}

		report("   Recording future use of analysis " << type
			<< " (" << use->second << " total) by " << newAnalysis->name);
	}
	
	// allocate dependencies
	for(auto& type : newAnalysis->required)
	{
		allocateDataStructure(uses, type, analyses, function, manager);
	}
}

static void freeDependencies(PassUseCountMap& uses,
	analysis::Analysis* newAnalysis)
{
	// decrement use count
	for(auto& analysisType : newAnalysis->required)
	{
		auto use = uses.find(analysisType);
		
		assert(use != uses.end());
		
		assert(use->second > 0);
		
		--use->second;
		
		report("   Recording user '" << newAnalysis->name
			<< "' finished with analysis " << analysisType
			<< " (" << use->second << " remaining)");
	}
}

static void allocateNewDataStructures(PassUseCountMap& uses,
	AnalysisMap& analyses, IRKernel* function, const Pass::StringVector& types,
	PassManager* manager)
{
	for(auto& analysisType : types)
	{
		auto use = uses.find(analysisType);
		
		assert(use != uses.end());
		
		assert(use->second > 0);
		
		--use->second;
		
		report(" Recording user (transform) finished with analysis "
			<< analysisType	<< " (" << use->second << " remaining)");
		
		allocateDataStructure(uses, analysisType, analyses, function, manager);
	}
}

static void runKernelPass(IRKernel* function, Pass* pass)
{
	report("  Running pass '" << pass->toString() << "' on function '"
		<< function->name << "'" );

	switch(pass->type)
	{
	case Pass::ImmutablePass:
	{
		assertM(false, "Immutable passes cannot be run on single functions.");
	}
	break;
	case Pass::ModulePass:
	{
		assertM(false, "Module passes cannot be run on single functions.");
	}
	break;
	case Pass::KernelPass:
	{
		KernelPass* functionPass = static_cast<KernelPass*>(pass);
		functionPass->runOnKernel(*function);
	}
	break;
	case Pass::ImmutableKernelPass:
	{
		ImmutableKernelPass* k = static_cast<ImmutableKernelPass*>(pass);
		k->runOnKernel(*function);
	}
	break;
	case Pass::BasicBlockPass:
	{
		BasicBlockPass* bbPass = static_cast<BasicBlockPass*>(pass);
		bbPass->initialize(*function);
		for(auto block = function->cfg()->begin(); 
			block != function->cfg()->end(); ++block)
		{
			bbPass->runOnBlock(*block);
		}
		bbPass->finalizeKernel();
	}
	break;
	case Pass::InvalidPass: assertM(false, "Invalid pass type.");
	}
}

static void runKernelPass(Module* module, IRKernel* function, Pass* pass)
{
	switch(pass->type)
	{
	case Pass::ImmutablePass: /* fall through */
	case Pass::ModulePass:
	break;
	case Pass::KernelPass:
	{
		report("  Running function pass '" << pass->toString()
			<< "' on function '"
			<< function->name << "'" );
		KernelPass* functionPass = static_cast<KernelPass*>(pass);
		functionPass->runOnKernel(*function);
	}
	break;
	case Pass::ImmutableKernelPass:
	{
		report("  Running immutable function pass '" << pass->toString()
			<< "' on function '" << function->name << "'" );
		ImmutableKernelPass* k = static_cast<ImmutableKernelPass*>(pass);
		k->runOnKernel(*function);
	}
	break;
	case Pass::BasicBlockPass:
	{
		report("  Running basic block pass '" << pass->toString() 
			<< "' on function '" << function->name << "'" );
		BasicBlockPass* bbPass = static_cast<BasicBlockPass*>(pass);
		bbPass->initialize(*function);
		for(auto block = function->cfg()->begin(); 
			block != function->cfg()->end(); ++block)
		{
			bbPass->runOnBlock(*block);
		}
		bbPass->finalizeKernel();
	}
	break;
	case Pass::InvalidPass: assertM(false, "Invalid pass type.");
	}
}

static void initializeKernelPass(Module* module, Pass* pass)
{
	switch(pass->type)
	{
	case Pass::ImmutablePass: /* fall through */
	case Pass::ModulePass:
	break;
	case Pass::KernelPass:
	{
		report("  Initializing function pass '" << pass->toString() << "'" );
		KernelPass* functionPass = static_cast<KernelPass*>(pass);
		functionPass->initialize(*module);
	}
	break;
	case Pass::ImmutableKernelPass:
	{
		report("  Initializing immutable function pass '"
			<< pass->toString() << "'" );
		ImmutableKernelPass* k = static_cast<ImmutableKernelPass*>(pass);
		k->initialize(*module);
	}
	break;
	case Pass::BasicBlockPass:
	{
		report("  Initializing basic block pass '" << pass->toString() << "'" );
		BasicBlockPass* bbPass = static_cast<BasicBlockPass*>(pass);
		bbPass->initialize(*module);
	}
	break;
	case Pass::InvalidPass: assertM(false, "Invalid pass type.");
	}
}

static void finalizeKernelPass(Module* module, Pass* pass)
{
	switch(pass->type)
	{
	case Pass::ImmutablePass: /* fall through */
	case Pass::ModulePass:
	break;
	case Pass::KernelPass:
	{
		report("  Finalizing function pass '" << pass->toString() << "'" );
		KernelPass* functionPass = static_cast<KernelPass*>(pass);
		functionPass->finalize();
	}
	break;
	case Pass::ImmutableKernelPass:
	{
		report("  Finalizing immutable function pass '"
			<< pass->toString() << "'" );
		ImmutableKernelPass* k = static_cast<ImmutableKernelPass*>(pass);
		k->finalize();
	}
	break;
	case Pass::BasicBlockPass:
	{
		report("  Finalizing basic block pass '" << pass->toString() << "'" );
		BasicBlockPass* bbPass = static_cast<BasicBlockPass*>(pass);
		bbPass->finalize();
	}
	break;
	case Pass::InvalidPass: assertM(false, "Invalid pass type.");
	}
}

static void runModulePass(Module* module, Pass* pass)
{
	report("  Running module pass '" << pass->toString() << "'" );
	switch(pass->type)
	{
	case Pass::ImmutablePass:
	{
		ImmutablePass* immutablePass = static_cast<ImmutablePass*>(pass);
		immutablePass->runOnModule(*module);
	}
	break;
	case Pass::ModulePass:
	{
		ModulePass* modulePass = static_cast<ModulePass*>(pass);
		modulePass->runOnModule(*module);
	}
	break;
	case Pass::KernelPass:     /* fall through */
	case Pass::BasicBlockPass: /* fall through */
	case Pass::ImmutableKernelPass:
	break;
	case Pass::InvalidPass: assertM(false, "Invalid pass type.");
	}
}

PassManager::PassManager(Module* module) :
	_module(module), _analyses(0)
{
	assert(_module != 0);
}

PassManager::~PassManager()
{
	clear();
}

void PassManager::addPass(Pass* pass)
{
	report("Adding pass '" << pass->toString() << "'");
	_passes.push_back(pass);
	pass->setPassManager(this);
}

void PassManager::addDependence(const std::string& dependentPassName,
	const std::string& passName)
{
	report("Adding dependency '" << dependentPassName
		<< "' <- '" << passName << "'");
	_extraDependences.insert(std::make_pair(dependentPassName, passName));
}

void PassManager::clear()
{
	for(auto pass = _passes.begin(); pass != _passes.end(); ++pass)
	{
		delete *pass;
	}
	
	_ownedTemporaryPasses.clear();
	_passes.clear();
	_extraDependences.clear();
}

void PassManager::releasePasses()
{	
	for(auto pass = _ownedTemporaryPasses.begin();
		pass != _ownedTemporaryPasses.end(); ++pass)
	{
		delete *pass;
	}
	
	_ownedTemporaryPasses.clear();
	_passes.clear();
	_extraDependences.clear();
}

void PassManager::runOnKernel(const std::string& name)
{
	auto function = _module->getKernel(name);

	runOnKernel(*function);
}

void PassManager::runOnKernel(IRKernel& function)
{
	report("Running pass manager on function " << function.name);

	PassWaveList passes = _schedulePasses();
	
	PassUseCountMap passesUseCounts = getPassUseCounts(passes);
	
	for(auto wave = passes.begin(); wave != passes.end(); ++wave)
	{
		for(auto pass = wave->begin(); pass != wave->end(); ++pass)
		{
			initializeKernelPass(_module, *pass);
		}
	
		AnalysisMap analyses;
	
		_analyses = &analyses;
		_function = &function;
	
		for(auto pass = wave->begin(); pass != wave->end(); ++pass)
		{
			allocateNewDataStructures(passesUseCounts, analyses,
				&function, (*pass)->analyses, this);
		
			runKernelPass(&function, *pass);
			_previouslyRunPasses[(*pass)->name] = *pass;
			
			freeUnusedDataStructures(passesUseCounts, analyses);
		}

		for(auto pass = wave->begin(); pass != wave->end(); ++pass)
		{
			finalizeKernelPass(_module, *pass);
		}

		_analyses = 0;
		_function = 0;
	}
	
	assert(passesUseCounts.empty());
	_previouslyRunPasses.clear();
}

void PassManager::runOnModule()
{
	report("Running pass manager on module " << _module->path());

	typedef std::map<std::string, AnalysisMap> AnalysisMapMap;
	
	AnalysisMapMap functionAnalyses;
	
	PassWaveList passes = _schedulePasses();

	PassUseCountMap passesUseCounts = getPassUseCounts(passes, *_module);
	
	// Run waves in order
	for(auto wave = passes.begin(); wave != passes.end(); ++wave)
	{
		// Run all module passes first
		for(auto pass = wave->begin(); pass != wave->end(); ++pass)
		{
			if((*pass)->type == Pass::KernelPass)   continue;
			if((*pass)->type == Pass::BasicBlockPass) continue;
		
			for(auto function = _module->kernels().begin();
				function != _module->kernels().end(); ++function)
			{
				auto analyses = functionAnalyses.insert(std::make_pair(
					function->first, AnalysisMap())).first;
				
				allocateNewDataStructures(passesUseCounts, analyses->second,
					function->second, (*pass)->analyses, this);
			
				freeUnusedDataStructures(passesUseCounts, analyses->second);
			}
			
			_previouslyRunPasses[(*pass)->name] = *pass;
			
			runModulePass(_module, *pass);
		}
	
		// Run all function and bb passes
		for(auto function = _module->kernels().begin();
			function != _module->kernels().end(); ++function)
		{
			for(auto pass = wave->begin(); pass != wave->end(); ++pass)
			{
				initializeKernelPass(_module, *pass);
			}
		
			auto analyses = functionAnalyses.insert(std::make_pair(
					function->first, AnalysisMap())).first;
				
			_analyses = &analyses->second;
			_function = function->second;
		
			for(auto pass = wave->begin(); pass != wave->end(); ++pass)
			{
				if((*pass)->type == Pass::ImmutablePass) continue;
				if((*pass)->type == Pass::ModulePass)    continue;
				
				allocateNewDataStructures(passesUseCounts, analyses->second,
					_function, (*pass)->analyses, this);
			
				runKernelPass(_module, _function, *pass);
				_previouslyRunPasses[(*pass)->name] = *pass;
			
				freeUnusedDataStructures(passesUseCounts, analyses->second);
			}

			for(auto pass = wave->begin(); pass != wave->end(); ++pass)
			{
				finalizeKernelPass(_module, *pass);
			}
		
			_analyses = 0;
			_function = 0;
		}
	}
	
	assert(passesUseCounts.empty());
	_previouslyRunPasses.clear();
}

PassManager::Analysis* PassManager::getAnalysis(const std::string& type)
{
	assert(_analyses != 0);

	AnalysisMap::iterator analysis = _analyses->find(type);
	if(analysis == _analyses->end())
	{
		analysis = _analyses->find(type);
	}
	
	if(analysis == _analyses->end()) return 0;
		
	return analysis->second;
}

const PassManager::Analysis* PassManager::getAnalysis(
	const std::string& type) const
{
	assert(_analyses != 0);

	AnalysisMap::const_iterator analysis = _analyses->find(type);
	if(analysis == _analyses->end()) return 0;
	
	return analysis->second;
}

void PassManager::invalidateAnalysis(const std::string& type)
{
	assert(_analyses != 0);

	AnalysisMap::iterator analysis = _analyses->find(type);
	if(analysis != _analyses->end())
	{
		report("Invalidating analysis " << type);
		delete analysis->second;
		_analyses->erase(analysis);
	}
}

void PassManager::invalidateAllAnalyses()
{
	assert(_analyses != 0);

	while(!_analyses->empty())
	{
		AnalysisMap::iterator analysis = _analyses->begin();
		report("Invalidating analysis " << analysis->first);
		delete analysis->second;
		_analyses->erase(analysis);
	}
}

Pass* PassManager::getPass(const std::string& name)
{
	auto pass = _previouslyRunPasses.find(name);
	if(pass != _previouslyRunPasses.end()) return pass->second;
	
	for(auto pass : _previouslyRunPasses)
	{
		if(pass.second->name == name)
		{
			return pass.second;
		}
	}
	
	return nullptr;
}

const Pass* PassManager::getPass(const std::string& name) const
{
	auto pass = _previouslyRunPasses.find(name);
	if(pass != _previouslyRunPasses.end()) return pass->second;
	
	for(auto pass : _previouslyRunPasses)
	{
		if(pass.second->name == name)
		{
			return pass.second;
		}
	}
	
	return nullptr;
}

PassManager::PassWaveList PassManager::_schedulePasses()
{
	typedef std::map<std::string, Pass*> PassMap;
	
	report(" Scheduling passes...");
	
	PassMap unscheduled;
	PassMap needDependencyCheck;
	
	report("  Initial list:");
	for(auto pass = _passes.begin(); pass != _passes.end(); ++pass)
	{
		report("   " << (*pass)->name);
		unscheduled.insert(std::make_pair((*pass)->name, *pass));
		needDependencyCheck.insert(std::make_pair((*pass)->name, *pass));
	}
	
	report("  Adding dependent passes:");
	while(!needDependencyCheck.empty())
	{
		auto pass = needDependencyCheck.begin();

		report("   for pass '" << pass->first << "'");
		
		auto dependentPasses = _getAllDependentPasses(pass->second);
		
		needDependencyCheck.erase(pass);
		
		for(auto dependentPass = dependentPasses.begin();
			dependentPass != dependentPasses.end(); ++dependentPass)
		{
			if(unscheduled.count(*dependentPass) == 0)
			{
				report("    adding '" << *dependentPass << "'");
				auto newPass = PassFactory::createPass(*dependentPass);
				addPass(newPass);
				_ownedTemporaryPasses.push_back(newPass);
				unscheduled.insert(std::make_pair(*dependentPass, newPass));
				needDependencyCheck.insert(
					std::make_pair(*dependentPass, newPass));
			}
		}
	}
	
	// Create waves by splitting transitions between different pass types
	//  in the dependence graph
	PassWaveList scheduled;
	
	PassMap unscheduledInWaves = unscheduled;

	report("  Setting up waves:");
	while(!unscheduledInWaves.empty())
	{
		report("   Wave: " << scheduled.size());
		scheduled.push_back(PassVector());
		
		for(auto pass = unscheduledInWaves.begin();
			pass != unscheduledInWaves.end(); )
		{
			bool unscheduledPredecessorsTransition = false;
			
			auto dependentPasses = _getAllDependentPasses(pass->second);
			
			report("   checking pass '" << pass->first << "'");
								
			for(auto dependentPassName = dependentPasses.begin();
				dependentPassName != dependentPasses.end(); ++dependentPassName)
			{
				if(unscheduledInWaves.count(*dependentPassName) != 0)
				{
					report("    would violate dependency '"
						<< *dependentPassName << "'");
								
					unscheduledPredecessorsTransition = true;
					break;
				}
			}
			
			if(!unscheduledPredecessorsTransition)
			{
				report("    adding '" << pass->first << "'");
				scheduled.back().push_back(pass->second);
				unscheduledInWaves.erase(pass++);
				continue;
			}
			
			++pass;
		}
		
		if(scheduled.back().empty())
		{
			throw std::runtime_error("Passes have circular dependencies!");
		}
	}
	
	// TODO sort unscheduled passes by weight
	
	report("  Final schedule:");
	
	for(auto wave = scheduled.begin(); wave != scheduled.end(); ++wave)
	{
		report("   Wave " << std::distance(scheduled.begin(), wave));
				
		PassVector newOrder;
		PassMap    unscheduledInThisWave;

		for(auto pass = wave->begin(); pass != wave->end(); ++pass)
		{
			unscheduledInThisWave.insert(std::make_pair((*pass)->name, *pass));		
		}
		
		while(!unscheduledInThisWave.empty())
		{
			bool scheduledAny = false;
			
			for(auto pass = unscheduledInThisWave.begin();
				pass != unscheduledInThisWave.end(); )
			{
				auto dependentPasses = _getAllDependentPasses(pass->second);
			
				bool dependenciesSatisfied = true;
			
				for(auto dependentPassName = dependentPasses.begin();
					dependentPassName != dependentPasses.end();
					++dependentPassName)
				{
					if(unscheduled.count(*dependentPassName) != 0)
					{
						dependenciesSatisfied = false;
						break;
					}
				}
			
				if(dependenciesSatisfied)
				{
					report("    " << pass->first);
					newOrder.push_back(pass->second);

					auto unscheduledPass = unscheduled.find(pass->first);
					assert(unscheduledPass != unscheduled.end());
					
					unscheduled.erase(unscheduledPass);

					unscheduledInThisWave.erase(pass++);
					scheduledAny = true;
					continue;
				}
				
				++pass;
			}

			if(!scheduledAny)
			{
				throw std::runtime_error("Passes have circular dependencies!");
			}
		}
		
		*wave = newOrder;
	}
	
	report("  Finished scheduling");
	
	return scheduled;
}

Pass::StringVector PassManager::_getAllDependentPasses(Pass* pass)
{
	Pass::StringVector dependentPasses = pass->getDependentPasses();
		
	auto extraDependences = _extraDependences.equal_range(pass->name);

	for(auto dependentPass = extraDependences.first;
		dependentPass != extraDependences.second; ++dependentPass)
	{
		dependentPasses.push_back(dependentPass->second);
	}
	
	return dependentPasses;
}

Pass* PassManager::_findPass(const std::string& name)
{
	for(auto pass = _passes.begin(); pass != _passes.end(); ++pass)
	{
		if((*pass)->name == name) return *pass;
	}
	
	assertM(false, "No pass named " << name);
	
	return 0;
}

}


