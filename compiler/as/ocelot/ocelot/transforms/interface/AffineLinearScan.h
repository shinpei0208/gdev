/*! \file AffineRegisterAllocationPass.cpp
	\author Rafael Martins de Souza <rafaelms@dcc.ufmg.br>
	\date Wednesday October 12, 2011
	\brief The header file for the AffineRegisterAllocationPass class.
*/

#ifndef AFFINE_LINEAR_SCAN_H_
#define AFFINE_LINEAR_SCAN_H_

// Standard Library Includes
// Ocelot Includes
#include <ocelot/transforms/interface/LinearScanRegisterAllocationPass.h>
#include <ocelot/analysis/interface/AffineAnalysis.h>

namespace transforms
{

class AffineLinearScan: public LinearScanRegisterAllocationPass
{
private:
	static unsigned MAX_WARPS;

	analysis::AffineAnalysis& _afa();
	virtual void _clear();
	virtual void _spill();
	virtual void _extendStack();
	virtual void _addCoalesced(const RegisterId id,
		const analysis::DataflowGraph::Type type);
	virtual void _coalesce();

	CoalescedArray _shared;
	const ir::Module* _m;

public:
	AffineLinearScan(unsigned registers = 8);
	virtual void initialize(const ir::Module& m);
	virtual void runOnKernel(ir::IRKernel& k);
	virtual void finalize();
};

}

#endif

