/*! \file   IPDOMReconvergencePass.h
	\date   Monday May 9, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the IPDOMReconvergence class.
*/

#ifndef IPDOM_RECONVERGENCE_PASS_H_INCLUDED
#define IPDOM_RECONVERGENCE_PASS_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/transforms/interface/Pass.h>

// Standard Library Includes
#include <vector>

namespace transforms
{

/*! \brief A pass to construct an instruction vector with reconverge points */
class IPDOMReconvergencePass : public KernelPass
{
public:
	typedef std::vector<ir::PTXInstruction> PTXInstructionVector;

public:
	/*! \brief The constructor sets the type */
	IPDOMReconvergencePass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();
	
public:
	PTXInstructionVector instructions;
};

}

#endif

