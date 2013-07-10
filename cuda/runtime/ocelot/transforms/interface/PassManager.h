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

//! Forward Declarations
class Pass;

/*! \brief A class to orchestrate the execution of many passes */
class PassManager
{
public:
	/*! \brief A map from analysis id to an up to date copy */
	typedef std::unordered_map<int, analysis::Analysis*> AnalysisMap;

public:
	/*! \brief The constructor creates an empty pass manager associated
		with an existing Module.  
		
		The module is not owned by the PassManager.
		
		\param module The module that this manager is associated with.
	*/
	explicit PassManager(ir::Module* module);
	~PassManager();
		
public:
	/*! \brief Adds a pass that needs to be eventually run
	
		The pass is not owned by the manager and must not be deallocated
		before it is run by the manager.
	
		\param pass The pass being added
	 */
	void addPass(Pass& pass);
	
	/*! \brief Adds an explicit dependence between pass types
	
		The dependence relationship is:
		
			dependentPassName <- passName
			
		or:
			
			dependentPassName depends on passName
	 */
	void addDependence(const std::string& dependentPassName,
		const std::string& passName);
	
	/*! \brief Clears all added passes */
	void clear();
	
	/*! \brief Deletes all added passes */
	void destroyPasses();
	
public:
	/*! \brief Runs passes on a specific Kernel contained in the module.
	
		\param name The name of the kernel to run all passes on.
	*/
	void runOnKernel(const std::string& name);

	/*! \brief Runs passes on a specific Kernel.
	
		\param kernel The kernel to run all passes on.
	*/
	void runOnKernel(ir::IRKernel& kernel);
	
	/*! \brief Runs passes on the entire module. */
	void runOnModule();

public:
	/*! \brief Get an up to date analysis by type */
	analysis::Analysis* getAnalysis(int type);

	/*! \brief Get an up to date analysis by type (const) */
	const analysis::Analysis* getAnalysis(int type) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again the next time 'get' is called */
	void invalidateAnalysis(int type);

public:
	/*! \brief Get a previously run pass by name */
	Pass* getPass(const std::string& name);

	/*! \brief Get a previously run pass by name (const) */
	const Pass* getPass(const std::string& name) const;

public:
	// Come on MSVS, get your act together!		
	#if !defined(_MSC_VER)
	/*! \brief Disallow the copy constructor */
	PassManager(const PassManager&) = delete;
	/*! \brief Disallow the assignment operator */
	const PassManager& operator=(const PassManager&) = delete;
	#endif

private:
	typedef std::multimap<int, Pass*, std::greater<int>> PassMap;
	typedef std::multimap<std::string, std::string> DependenceMap;
	typedef std::unordered_map<std::string, Pass*> PassNameMap;
	typedef std::vector<Pass*> PassVector;
	typedef std::list<PassVector> PassWaveList;
	typedef std::vector<std::string> StringVector;

private:
	PassWaveList _schedulePasses();
	StringVector _getAllDependentPasses(Pass* p);
	Pass* _findPass(const std::string& name);

private:
	PassMap       _passes;
	ir::Module*   _module;
	ir::IRKernel* _kernel;
	AnalysisMap*  _analyses;
	PassVector    _ownedTemporaryPasses;
	PassNameMap   _previouslyRunPasses;
	DependenceMap _extraDependences;
};

}

#endif

