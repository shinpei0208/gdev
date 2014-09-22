/*! \file PassManager.h
	\date Thursday September 16, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the PassManager class
*/

#ifndef PASS_MANAGER_H_INCLUDED
#define PASS_MANAGER_H_INCLUDED

// Standard Library Includes
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <list>

// Forward Declarations
namespace analysis   { class Analysis; }
namespace ir         { class Module;   }
namespace ir         { class IRKernel; }
namespace transforms { class Pass;     }

namespace transforms
{

/*! \brief A class to orchestrate the execution of many passes */
class PassManager
{
public:
	typedef analysis::Analysis Analysis;
	typedef ir::Module         Module;
	typedef ir::IRKernel       IRKernel;

public:
	/*! \brief A map from analysis id to an up to date copy */
	typedef std::unordered_map<std::string, Analysis*> AnalysisMap;
	
	typedef std::vector<Pass*> PassVector;
	typedef std::list<PassVector> PassWaveList;

public:
	/*! \brief The constructor creates an empty pass manager associated
		with an existing Module.  
		
		The module is not owned by the PassManager.
		
		\param module The module that this manager is associated with.
	*/
	explicit PassManager(Module* module);
	~PassManager();
		
public:
	/*! \brief Adds a pass that needs to be eventually run
	
		The pass is now owned by the manager.
	
		\param pass The pass being added
	 */
	void addPass(Pass* pass);
	
	/*! \brief Adds an explicit dependence between pass types
	
		The dependence relationship is:
		
			dependentPassName <- passName
			
		or:
			
			dependentPassName depends on passName
	 */
	void addDependence(const std::string& dependentPassName,
		const std::string& passName);
	
	/*! \brief Clears all added passes, deleting them */
	void clear();

	/*! \brief Releases all added passes, they are no longer
		owned by the manager */
	void releasePasses();
	
public:
	/*! \brief Runs passes on a specific IRKernel contained in the module.
	
		\param name The name of the function to run all passes on.
	*/
	void runOnKernel(const std::string& name);

	/*! \brief Runs passes on a specific IRKernel.
	
		\param function The function to run all passes on.
	*/
	void runOnKernel(IRKernel& function);
	
	/*! \brief Runs passes on the entire module. */
	void runOnModule();

public:
	/*! \brief Get an up to date analysis by type */
	Analysis* getAnalysis(const std::string& type);

	/*! \brief Get an up to date analysis by type (const) */
	const Analysis* getAnalysis(const std::string& type) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again the next time 'get' is called */
	void invalidateAnalysis(const std::string& type);
	
	/*! \brief Invalidate all analyses, the pass manager will
		need to generate them again the next time 'get' is called */
	void invalidateAllAnalyses();

public:
	/*! \brief Get a previously run pass by name */
	Pass* getPass(const std::string& name);

	/*! \brief Get a previously run pass by name (const) */
	const Pass* getPass(const std::string& name) const;

public:	
	/*! \brief Disallow the copy constructor */
	PassManager(const PassManager&) = delete;
	/*! \brief Disallow the assignment operator */
	const PassManager& operator=(const PassManager&) = delete;

private:
	typedef std::multimap<std::string, std::string> DependenceMap;
	typedef std::unordered_map<std::string, Pass*> PassMap;
	typedef std::vector<std::string> StringVector;

private:
	PassWaveList _schedulePasses();
	StringVector _getAllDependentPasses(Pass* p);
	Pass*        _findPass(const std::string& name);

private:
	PassVector    _passes;
	Module*       _module;
	IRKernel*     _function;
	AnalysisMap*  _analyses;
	PassVector    _ownedTemporaryPasses;
	DependenceMap _extraDependences;
	PassMap       _previouslyRunPasses;
};

}

#endif

