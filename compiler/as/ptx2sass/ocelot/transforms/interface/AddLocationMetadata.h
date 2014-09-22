/*! \file   AddLocationMetadataPass.h
	\date   Wednesday March 14, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the  AddScalarMetadataPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

namespace transforms
{

class AddLocationMetadataPass : public ::transforms::KernelPass
{
public:
	/*! \brief The default constructor sets the type */
	AddLocationMetadataPass();
	
public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);
	/*! \brief Finalize the pass */
	void finalize();
};

}

