/*! \file   DefaultLayoutPass.h
	\author Gregory Diamso <gregory.diamos@gatech.edu>
	\date   Monday May 16, 2011
	\brief  The header file for the DefaultLayoutPass class.
*/

#ifndef DEFAULT_LAYOUT_PASS_H_INCLUDED
#define DEFAULT_LAYOUT_PASS_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/transforms/interface/Pass.h>

// Standard Library Includes
#include <vector>

namespace transforms
{

/*! \brief Construct an instruction vector without reconverge points */
class DefaultLayoutPass : public ImmutableKernelPass
{
public:
	typedef std::vector<ir::PTXInstruction> PTXInstructionVector;

public:
	/*! \brief The constructor sets the type */
	DefaultLayoutPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(const ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();
	
public:
	PTXInstructionVector instructions;
};

}

#endif

