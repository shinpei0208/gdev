/*!	\file   ModuleLinkerPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday December 4, 2012
	\brief  The header file for the ModuleLinkerPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

/*! \brief A transform to link multiple PTX modules together */
class ModuleLinkerPass : public ModulePass
{
public:
	typedef std::vector<std::string> StringVector;

public:
	/*! \brief Create the pass, create dependencies */
	ModuleLinkerPass(bool modifyModuleInPlace = false);
	~ModuleLinkerPass();

public:
	/*! \brief Run the pass on a specific module */
	void runOnModule(ir::Module& m);

public:
	ir::Module* linkedModule() const;
	StringVector getAllUndefinedSymbols() const;

public:
	StringVector getAllSymbolsUsedByKernel(const std::string& kernelName) const;
	void deleteAllSymbolsExceptThese(const StringVector& symbolsToKeep);

public:
	#ifndef _WIN32
	ModuleLinkerPass(const ModuleLinkerPass&) = delete;
	ModuleLinkerPass& operator=(const ModuleLinkerPass&) = delete;
	#endif

public:
	void _linkTextures(ir::Module& m);
	void _linkGlobals(ir::Module& m);
	void _linkFunctions(ir::Module& m);
	void _linkPrototypes(ir::Module& m);

private:
	ir::Module* _linkedModule;
	bool        _inPlace;
};


}

