/*!	\file   ModuleLinkerPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday December 4, 2012
	\brief  The source file for the ModuleLinkerPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/ModuleLinkerPass.h>

#include <ocelot/ir/interface/Module.h>

#include <ocelot/api/interface/ocelot.h>

// Standard Library Includes
#include <stdexcept>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

ModuleLinkerPass::ModuleLinkerPass(bool inPlace)
: ModulePass({}, "ModuleLinkerPass"), _linkedModule(0),
	_inPlace(inPlace)
{
	
}

ModuleLinkerPass::~ModuleLinkerPass()
{
	if(!_inPlace) delete _linkedModule;
}

void ModuleLinkerPass::runOnModule(ir::Module& m)
{
	report("Linking module " << m.path());
	
	if(_linkedModule == 0)
	{
		if(_inPlace)
		{
			_linkedModule = &m;
			
			return;
		}
		else
		{
			_linkedModule = new ir::Module;
		
			_linkedModule->isLoaded();
		}
	}

	_linkTextures(m);
	_linkGlobals(m);
	_linkFunctions(m);
	_linkPrototypes(m);
}

ir::Module* ModuleLinkerPass::linkedModule() const
{
	return _linkedModule;
}

bool containsSymbol(ir::Module& module, ir::PTXKernel& kernel,
	const std::string& symbol)
{
	if(kernel.parameters.count(symbol) != 0) return true;
	    if(kernel.locals.count(symbol) != 0) return true;

	for(auto argument = kernel.arguments.begin();
		argument != kernel.arguments.end(); ++argument)
	{
		if(argument->name == symbol) return true;
	}
	
	 if(module.kernels().count(symbol) != 0) return true;
	if(module.textures().count(symbol) != 0) return true;
	 if(module.globals().count(symbol) != 0) return true;
	
	if(ocelot::isExternalFunction(symbol)) return true;
	
	// built-in
	if(symbol == "cudaLaunchDevice")      return true;
	if(symbol == "cudaDeviceSynchronize") return true;
	
	return false;
}

typedef std::set<std::string>    StringSet;
typedef std::vector<std::string> StringVector;
	
static StringVector getAllSymbolsUsedByThisKernel(
	const std::string& kernelName, ir::Module* module)
{
	auto kernel = module->kernels().find(kernelName);

	if(kernel == module->kernels().end()) return StringVector();
	
	StringSet encountered;
	
	for(auto block = kernel->second->cfg()->begin();
		block != kernel->second->cfg()->end(); ++block)
	{
		for(auto instruction = block->instructions.begin();
			instruction != block->instructions.end(); ++instruction)
		{
			typedef std::vector<ir::PTXOperand*> OperandVector;
			
			auto ptx = static_cast<ir::PTXInstruction*>(*instruction);
		
			OperandVector operands;
			
			operands.push_back(&ptx->a);
			operands.push_back(&ptx->b);
			operands.push_back(&ptx->pg);
			operands.push_back(&ptx->pq);
			operands.push_back(&ptx->d);
			
			if(ptx->opcode != ir::PTXInstruction::Call)
			{
				 operands.push_back(&ptx->c);
			}
		
			for(auto operand = operands.begin();
				operand != operands.end(); ++operand)
			{
				if((*operand)->addressMode != ir::PTXOperand::Address &&
					(*operand)->addressMode != ir::PTXOperand::FunctionName)
				{
					continue;
				}
				
				encountered.insert((*operand)->identifier);
			}
		}
	}
	
	return StringVector(encountered.begin(), encountered.end());
}
	
ModuleLinkerPass::StringVector ModuleLinkerPass::getAllUndefinedSymbols() const
{
	StringVector undefined;
	
	if(_linkedModule == 0) return undefined;
	
	StringSet encountered;
	
	for(auto kernel = _linkedModule->kernels().begin();
		kernel != _linkedModule->kernels().end(); ++kernel)
	{
		auto symbolsUsedByKernel = getAllSymbolsUsedByThisKernel(kernel->first,	
			_linkedModule);
		
		for(auto symbol = symbolsUsedByKernel.begin();
			symbol != symbolsUsedByKernel.end(); ++symbol)
		{
			if(!encountered.insert(*symbol).second) continue;
			
			if(!containsSymbol(*_linkedModule, *kernel->second, *symbol))
			{
				undefined.push_back(*symbol);
			}
		}
	}
	
	return undefined;
}

static bool isKernelSymbol(ir::Module* module,
	const std::string& symbol)
{
	return module->kernels().count(symbol) != 0;
}

ModuleLinkerPass::StringVector ModuleLinkerPass::getAllSymbolsUsedByKernel(
	const std::string& kernelName) const
{
	StringSet usedSymbols;

	usedSymbols.insert(kernelName);

	StringVector unprocessedSymbols = getAllSymbolsUsedByThisKernel(
		kernelName, _linkedModule);
	
	while(!unprocessedSymbols.empty())
	{
		StringVector newSymbols;
	
		for(auto symbol = unprocessedSymbols.begin();
			symbol != unprocessedSymbols.end(); ++symbol)
		{
			if(!usedSymbols.insert(*symbol).second) continue;
		
			if(!isKernelSymbol(_linkedModule, *symbol)) continue;
		
			StringVector kernelSymbols = getAllSymbolsUsedByThisKernel(
				*symbol, _linkedModule);
				
			newSymbols.insert(newSymbols.end(), kernelSymbols.begin(),
				kernelSymbols.end());
		}
		
		unprocessedSymbols = std::move(newSymbols);
	}
	
	return StringVector(usedSymbols.begin(), usedSymbols.end());
}

static void removeSymbol(ir::Module* module, const std::string& symbol)
{
	if(module->kernels().count(symbol) != 0)
	{
		module->removeKernel(symbol);
	}
	else if(module->textures().count(symbol) != 0)
	{
		module->removeTexture(symbol);
	}
	else if(module->globals().count(symbol) != 0)
	{
		module->removeGlobal(symbol);
	}
	else if(module->prototypes().count(symbol) != 0)
	{
		module->removePrototype(symbol);
	}
}

static StringVector getAllSymbols(ir::Module* module)
{
	StringVector symbols;
	
	for(auto kernel = module->kernels().begin();
		kernel != module->kernels().end(); ++kernel)
	{
		symbols.push_back(kernel->first);
	}
	
	for(auto prototype = module->prototypes().begin();
		prototype != module->prototypes().end(); ++prototype)
	{
		symbols.push_back(prototype->first);
	}
	
	for(auto global = module->globals().begin();
		global != module->globals().end(); ++global)
	{
		symbols.push_back(global->first);
	}
	
	for(auto texture = module->textures().begin();
		texture != module->textures().end(); ++texture)
	{
		symbols.push_back(texture->first);
	}
	
	return symbols;
}

void ModuleLinkerPass::deleteAllSymbolsExceptThese(
	const StringVector& symbolsToKeep)
{
	StringSet keptSymbols(symbolsToKeep.begin(), symbolsToKeep.end());
	
	auto symbols = getAllSymbols(_linkedModule);
	
	for(auto symbol = symbols.begin();
		symbol != symbols.end(); ++symbol)
	{
		if(keptSymbols.count(*symbol) != 0) continue;
		
		removeSymbol(_linkedModule, *symbol);
	}
}

void ModuleLinkerPass::_linkTextures(ir::Module& m)
{
	report(" Linking textures...");
	for(auto texture = m.textures().begin();
		texture != m.textures().end(); ++texture)
	{
		report("  " << texture->first);
		
		if(_linkedModule->textures().count(texture->first))
		{
			throw std::runtime_error(
				"No support for duplicate textures YET (needed for '" +
				texture->first + "')");
		}
		
		_linkedModule->insertTexture(texture->second);
	}
}

static bool isNumeric(char c)
{
	return c == '0' || c == '1' || c == '2' || c == '3' || c == '4' ||
		c == '5' || c == '6' || c == '7' || c == '8' || c == '9';
}

static std::string findNumericSuffix(ir::Module& module,
	const std::string& base, unsigned int begin)
{
	while(true)
	{
		std::stringstream stream;
		
		stream << base << begin++;
		
		if(module.globals().count(stream.str()) == 0 &&
			module.kernels().count(stream.str()) == 0 &&
			module.textures().count(stream.str()) == 0)
		{
			return stream.str();
		}
	}
	
	assertM(false, "Could not find any valid identifier.");
	
	return base;
}

static void findSimpleUniqueName(ir::Module& module, std::string& name)
{
	assert(!name.empty());
	assert(!isNumeric(name[0]));
	
	if(!isNumeric(*name.rbegin())) return;
	
	size_t endPosition = name.size();
	
	for(; endPosition != 0; --endPosition)
	{
		if(!isNumeric(name[endPosition - 1])) break;
	}
	
	std::string base = name.substr(0, endPosition);
	
	name = findNumericSuffix(module, base, 0);
}

static void findComplexUniqueName(ir::Module& module, std::string& name)
{
	name = findNumericSuffix(module, name, 0);
}

static void replaceInstances(ir::Module& module, const std::string& oldName,
	const std::string& newName)
{
	for(auto kernel = module.kernels().begin();
		kernel != module.kernels().end(); ++kernel)
	{
		for(auto block = kernel->second->cfg()->begin();
			block != kernel->second->cfg()->end(); ++block)
		{
			for(auto instruction = block->instructions.begin();
				instruction != block->instructions.end(); ++instruction)
			{
				typedef std::vector<ir::PTXOperand*> OperandVector;
				
				auto ptx = static_cast<ir::PTXInstruction*>(*instruction);
			
				OperandVector operands;
				
				operands.push_back(&ptx->a);
				operands.push_back(&ptx->b);
				operands.push_back(&ptx->pg);
				operands.push_back(&ptx->pq);
				operands.push_back(&ptx->d);
				
				if(ptx->opcode != ir::PTXInstruction::Call)
				{
					 operands.push_back(&ptx->c);
				}
				
				for(auto operand = operands.begin();
					operand != operands.end(); ++operand)
				{
					if((*operand)->addressMode != ir::PTXOperand::Address)
					{
						continue;
					}
					
					if((*operand)->identifier == oldName)
					{
						(*operand)->identifier = newName;
					}
				}
			}
		}
	}
}

static std::string renameGlobal(ir::Module& module, const ir::Global& global)
{
	std::string newName = global.name();
	
	findSimpleUniqueName(module, newName);
	
	if(newName == global.name()) findComplexUniqueName(module, newName);
	
	report("   renaming to " << newName);
	
	replaceInstances(module, global.name(), newName);
	
	return newName;
}

static void handleDuplicateGlobal(ir::Module& module, const ir::Global& global)
{
	auto existingGlobal = module.globals().find(global.name());
	
	assert(existingGlobal != module.globals().end());

	bool bothPrivate =
		(global.statement.attribute == ir::PTXStatement::NoAttribute)
		&& (existingGlobal->second.statement.attribute ==
			ir::PTXStatement::NoAttribute);

	if(bothPrivate)
	{
		auto newName = renameGlobal(module, existingGlobal->second);
		
		ir::Global newGlobal      = global;
		ir::Global existingGlobal = *module.getGlobal(global.name());
		
		module.removeGlobal(global.name());
		
		existingGlobal.statement.name = newName;
		
		module.insertGlobal(newGlobal);
		module.insertGlobal(existingGlobal);
	}
	else
	{
		bool override =
			(existingGlobal->second.statement.attribute ==
				ir::PTXStatement::Weak)
			|| (existingGlobal->second.statement.attribute ==
				ir::PTXStatement::Extern);
				
		if(override)
		{
			report("   overriding with copy from new module.");
	
			module.removeGlobal(global.name());
			module.insertGlobal(global);
		}
		else
		{
			report("   overriding with existing copy.");
		}
	}
}

void ModuleLinkerPass::_linkGlobals(ir::Module& m)
{
	report(" Linking globals...");
	for(auto global = m.globals().begin();
		global != m.globals().end(); ++global)
	{
		report("  " << global->second.name());
		
		if(_linkedModule->globals().count(global->first))
		{
			handleDuplicateGlobal(*_linkedModule, global->second);
			continue;
		}
		
		_linkedModule->insertGlobal(global->second);
	}
}

static void handleDuplicateFunction(ir::Module& module,
	const ir::PTXKernel& function)
{
	auto existingFunction = module.kernels().find(function.name);
	
	assert(existingFunction != module.kernels().end());
	
	if(existingFunction->second->function() != function.function())
	{
		throw std::runtime_error(
			"Function and Kernel declared with the same name: '" +
			function.name + "'.");	
	}
	
	auto originalPrototype = existingFunction->second->getPrototype();
	auto      newPrototype =                  function.getPrototype();
	
	auto originalAttribute = originalPrototype.linkingDirective;
	auto      newAttribute =      newPrototype.linkingDirective;
	
	bool replace = (originalAttribute == ir::PTXKernel::Prototype::Visible ||
		ir::PTXKernel::Prototype::Weak) &&
		newAttribute == ir::PTXKernel::Prototype::Weak;
	
	if(replace)
	{
		module.removePrototype(function.name);
		module.removeKernel(function.name);

		module.insertKernel(new ir::PTXKernel(function));
		module.addPrototype(function.name, newPrototype);
	}
}

void ModuleLinkerPass::_linkFunctions(ir::Module& m)
{
	report(" Linking functions...");
	for(auto function = m.kernels().begin();
		function != m.kernels().end(); ++function)
	{
		report("  " << function->second->getPrototype().toString());
		
		if(_linkedModule->kernels().count(function->first))
		{
			handleDuplicateFunction(*_linkedModule, *function->second);
			continue;
		}
		
		_linkedModule->insertKernel(new ir::PTXKernel(*function->second));
		_linkedModule->addPrototype(function->second->name,
			function->second->getPrototype());
	}
}

static void handleDuplicatePrototype(ir::Module& module,
	const ir::PTXKernel::Prototype& prototype)
{
	auto existingPrototype = module.prototypes().find(prototype.identifier);
	
	assert(existingPrototype != module.prototypes().end());
		
	auto originalAttribute = existingPrototype->second.linkingDirective;
	auto      newAttribute =                 prototype.linkingDirective;
	
	bool replace = (originalAttribute == ir::PTXKernel::Prototype::Visible ||
		ir::PTXKernel::Prototype::Weak) &&
		newAttribute == ir::PTXKernel::Prototype::Weak;
	
	if(replace)
	{
		module.removePrototype(prototype.identifier);
		module.addPrototype(prototype.identifier, prototype);
	}
}

void ModuleLinkerPass::_linkPrototypes(ir::Module& m)
{
	report(" Linking prototypes...");
	for(auto prototype = m.prototypes().begin();
		prototype != m.prototypes().end(); ++prototype)
	{
		report("  " << prototype->second.toString());
		
		if(_linkedModule->prototypes().count(prototype->first))
		{
			handleDuplicatePrototype(*_linkedModule, prototype->second);
			continue;
		}
		
		_linkedModule->addPrototype(prototype->first, prototype->second);
	}
}

}

