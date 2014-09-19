/*! \file   SimplifyExternalCallsPass.h
	\date   Saturday April 9, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the SimplifyExternalCallsPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

// Forward Declarations
namespace ir { class ExternalFunctionSet; }

namespace transforms
{

/*! \brief Removes parameters passed to external calls to eliminate explicit
	stack modifications in PTX.  The register are passed directly to external
	calls. */
class SimplifyExternalCallsPass : public KernelPass
{
public:
	/*! \brief The constructor configures the pass
	
		\param simplifyAll Convert all parameter arguments to registers.
	 */
	SimplifyExternalCallsPass(const ir::ExternalFunctionSet&,
		bool simplifyAll = false);
		
	SimplifyExternalCallsPass();

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();

private:
	const ir::ExternalFunctionSet* _externals;
	bool                           _simplifyAll;
};

}

