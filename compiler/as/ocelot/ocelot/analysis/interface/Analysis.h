/*! \file   Analysis.h
	\author Gregory Diamos <gdiamos@gatech.edu>
	\date   Saturday May 7, 2011
	\brief  The header file for the Analysis class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace transforms { class PassManager; }
namespace ir         { class IRKernel;    }
namespace ir         { class Module;      }

namespace analysis
{

/*! \brief An analysis that can be constructed for aiding IR transforms */
class Analysis
{
public:
	typedef std::vector<std::string> StringVector;

public:
	/*! \brief Initialize the analysis, register it with a pass manager */
	Analysis(const std::string& name = "",
		const StringVector& required = StringVector());

	virtual ~Analysis();

public:
	virtual void configure(const StringVector& options);

public:
	/*! \brief The name of the analysis */
	const std::string name;

	/*! \brief The analysis dependencies */
	const StringVector required;

public:
	/*! \brief Set the pass manager used to supply dependent analyses */
	void setPassManager(transforms::PassManager* m);
	
	/*! \brief Get an up to date analysis by type */
	Analysis* getAnalysis(const std::string& name);

	/*! \brief Get an up to date analysis by type (const) */
	const Analysis* getAnalysis(const std::string& name) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again for other users */
	void invalidateAnalysis(const std::string& name);

private:
	transforms::PassManager* _manager;

};

/*! \brief An analysis over a single kernel */
class KernelAnalysis : public Analysis
{

public:
	/*! \brief Initialize the analysis, register it with a pass manager */
	KernelAnalysis(const std::string& name = "",
		const StringVector& required = StringVector());

public:
	virtual void analyze(ir::IRKernel& kernel) = 0;

};

/*! \brief An analysis over a complete module */
class ModuleAnalysis : public Analysis
{

public:
	/*! \brief Initialize the analysis, register it with a pass manager */
	ModuleAnalysis(const std::string& name = "",
		const StringVector& required = StringVector());

public:
	virtual void analyze(ir::Module& module) = 0;

};


}

