/*! \file SubkernelFormationPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday July 27, 2010
	\brief The header file for the SubkernelFormationPass class.
*/

#ifndef SUBKERNEL_FORMATION_PASS_H_INCLUDED
#define SUBKERNEL_FORMATION_PASS_H_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

// Standard Library Includes
#include <vector>
#include <unordered_map>

// Forward Declarations
namespace analysis { class DataflowGraph; }
namespace ir       { class PTXKernel;     }

namespace transforms
{

/*! \brief Split all kernels in a module into sub-kernels.  The sub-kernels
	should be called as functions from the main kernel.  The assumption
	is that all threads will execute a sub-kernel, hit a barrier, and
	enter the next sub-kernel.
	
	This pass may optionally insert an explicit scheduler kernel that is
	responsible for doing fine-grained scheduling of the next function
	to execute and control transition between functions.  This is necessary
	to support intelligent scheduling on architectures without runtime 
	support (mainly GPUs).
 */
class SubkernelFormationPass : public ModulePass
{
public:
	typedef std::vector<ir::PTXKernel*> KernelVector;
		
public:
	SubkernelFormationPass(unsigned int expectedRegionSize = 50);
	void runOnModule(ir::Module& m);

public:
	void setExpectedRegionSize(unsigned int s);

public:
	class ExtractKernelsPass : public KernelPass
	{
	public:
		typedef std::unordered_map<ir::PTXKernel*, 
			KernelVector> KernelVectorMap;
			
	public:
		ExtractKernelsPass(unsigned int expectedRegionSize = 50);
		void initialize(const ir::Module& m);
		void runOnKernel(ir::IRKernel& k);
		void finalize();
		
	public:
		KernelVectorMap kernels;
	
	private:
		analysis::DataflowGraph& _dfg();
	
	private:
		unsigned int _expectedRegionSize;
	};
	
private:
	unsigned int _expectedRegionSize;
};

}

#endif

