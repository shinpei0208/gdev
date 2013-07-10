/*! \file   MIMDThreadSchedulingPass.h
	\date   Friday February 18, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MIMDThreadSchedulingPass class.
*/

#ifndef MIMD_THREAD_SCHEDULING_PASS_H_INCLUDED
#define MIMD_THREAD_SCHEDULING_PASS_H_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{
/*! \brief A class for changing the scheduling order of threads assuming
	execution on a SIMT IPDOM machine. */
class MIMDThreadSchedulingPass : public KernelPass
{
public:
	/*! \brief The constructor sets the required analysis information */
	MIMDThreadSchedulingPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();

};


}

#endif

