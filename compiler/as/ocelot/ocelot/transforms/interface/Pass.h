/*! \file Pass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday September 15, 2009
	\brief The header file for the Pass class.
*/

#ifndef PASS_H_INCLUDED
#define PASS_H_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>

// Standard Library Includes
#include <string>
#include <list>
#include <vector>

// Forward Declarations
namespace ir
{
	class IRKernel;
	class Module;
	class BasicBlock;
}

namespace transforms
{
	class PassManager;
}

namespace transforms
{
/*! \brief A class modeled after the LLVM notion of an optimization pass.  
	Allows different transformations to be applied to PTX modules */
class Pass
{
public:
	/*! \brief For virtual classes, the type of pass */
	enum Type
	{
		ImmutablePass,
		ModulePass,
		KernelPass,
		ImmutableKernelPass,
		BasicBlockPass,
		InvalidPass
	};
	
	/*! \brief shorthand for analysis */
	typedef analysis::Analysis Analysis;
	
	typedef std::vector<std::string> StringVector; 
	
public:
	/*! \brief The type of this pass */
	const Type type;
	
	/*! \brief What types of analysis routines does the pass require? */
	const StringVector analyses;
	
	/*! \brief The name of the pass */
	const std::string name;

public:
	/*! \brief The default constructor sets the type */
	Pass(Type t = InvalidPass, const StringVector& required = StringVector(),
		const std::string& n = "");
	/*! \brief Virtual destructor */
	virtual ~Pass();

public:
	/*! \brief Set the pass manager used to supply dependent analyses */
	void setPassManager(transforms::PassManager* m);

	/*! \brief Get an up to date analysis by type */
	Analysis* getAnalysis(const std::string& name);

	/*! \brief Get an up to date analysis by type (const) */
	const Analysis* getAnalysis(const std::string& name) const;
	
	/*! \brief Invalidate the analysis, the pass manager will
		need to generate it again for other applications */
	void invalidateAnalysis(const std::string& name);
	
	/*! \brief Invalidate all analyses, the pass manager will
		need to generate it again for other applications */
	void invalidateAllAnalyses();

public:
	/*! \brief Get a list of passes that this pass depends on */
	virtual StringVector getDependentPasses() const;

public:
	/*! \brief Report the name of the pass */
	std::string toString() const;
	
private:
	PassManager* _manager;
};


/*! \brief A pass that generates information about a 
	program without modifying it, used to generate data structures */
class ImmutablePass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	ImmutablePass(const StringVector& required = StringVector(),
		const std::string& n = "");
	/*! \brief Virtual destructor */
	virtual ~ImmutablePass();
	
public:
	/*! \brief Run the pass on a specific module */
	virtual void runOnModule(const ir::Module& m) = 0;
};

/*! \brief A pass over an entire module */
class ModulePass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	ModulePass(const StringVector& required = StringVector(),
		const std::string& n = "");
	/*! \brief Virtual destructor */
	virtual ~ModulePass();
	
public:
	/*! \brief Run the pass on a specific module */
	virtual void runOnModule(ir::Module& m) = 0;		
};

/*! \brief A pass over a single kernel in a module */
class KernelPass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	KernelPass(const StringVector& required = StringVector(),
		const std::string& n = "");
	/*! \brief Virtual destructor */
	virtual ~KernelPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	virtual void runOnKernel(ir::IRKernel& k) = 0;		
	/*! \brief Finalize the pass */
	virtual void finalize();
};

/*! \brief An immutable pass over a single kernel in a module */
class ImmutableKernelPass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	ImmutableKernelPass(const StringVector& required = StringVector(),
		const std::string& n = "");
	/*! \brief Virtual destructor */
	virtual ~ImmutableKernelPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const ir::Module& m) = 0;
	/*! \brief Run the pass on a specific kernel in the module */
	virtual void runOnKernel(const ir::IRKernel& k) = 0;		
	/*! \brief Finalize the pass */
	virtual void finalize() = 0;
};

/*! \brief A pass over a single basic block in a kernel */
class BasicBlockPass : public Pass
{
public:
	/*! \brief The default constructor sets the type */
	BasicBlockPass(const StringVector& required = StringVector(),
		const std::string& n = "");
	/*! \brief Virtual destructor */
	virtual ~BasicBlockPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const ir::Module& m) = 0;
	/*! \brief Initialize the pass using a specific kernel */
	virtual void initialize(const ir::IRKernel& m) = 0;
	/*! \brief Run the pass on a specific kernel in the module */
	virtual void runOnBlock(ir::BasicBlock& b) = 0;		
	/*! \brief Finalize the pass on the kernel */
	virtual void finalizeKernel() = 0;
	/*! \brief Finalize the pass on the module */
	virtual void finalize() = 0;
};

typedef std::list<Pass*> PassList;

}

#endif

