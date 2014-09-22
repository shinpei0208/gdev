/*! \file   MIMDThreadSchedulingPass.h
	\date   Friday February 18, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MIMDThreadSchedulingPass class.
*/

#pragma once

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
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);

public:
	class Statistics
	{
	public:
		Statistics();
	
	public:
		unsigned int totalInstructions;
		unsigned int totalSafeInstructions;
	
	public:
		void reset();
	};

public:
	Statistics statistics;

};


}


