/*! \file SyncEliminationPass.h
	\date Aug 30, 2010
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\brief The header file for the SyncEliminationPass class
*/

#ifndef SYNCELIMINATIONPASS_H_
#define SYNCELIMINATIONPASS_H_

#include <ocelot/transforms/interface/Pass.h>
namespace transforms {
/*! \brief This pass converts ordinary bra instructions into bra.uni, whenever
    the divergence analysis deems it safe to do so.
 */
class SyncEliminationPass : public KernelPass
{
private:

public:
	SyncEliminationPass();
	virtual ~SyncEliminationPass() {};
	virtual void initialize( const ir::Module& m ){};
	virtual void runOnKernel( ir::IRKernel& k );
	virtual void finalize(){};
};

}

#endif /* BLOCKUNIFICATIONPASS_H_ */

