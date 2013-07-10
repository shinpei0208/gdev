/*! \file DivergenceLinearScan.h
	\author Diogo Nunes Sampaio <dnsampaio@gmail.com>
	\date Apr 5, 2012
	\brief The header file for the DivergenceLinearScan class.
*/
#ifndef DIVERGENCELINEARSCAN_H_
#define DIVERGENCELINEARSCAN_H_

// Ocelot Includes
#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/transforms/interface/LinearScanRegisterAllocationPass.h>

namespace transforms {

class DivergenceLinearScan: public LinearScanRegisterAllocationPass
{
  public:
    DivergenceLinearScan(unsigned registers = 8);
    ~DivergenceLinearScan();
    virtual void initialize(const ir::Module& m);
    virtual void runOnKernel(ir::IRKernel& k);
    virtual void finalize();
    virtual void _spill();


  private:
    CoalescedArray _shared;
    const ir::Module* _m;
    analysis::DivergenceAnalysis& _diva();
    virtual void _clear();
    virtual void _extendStack();
    virtual void _addCoalesced(const RegisterId id,
    	const analysis::DataflowGraph::Type type);
    virtual void _coalesce();
};

}

#endif

