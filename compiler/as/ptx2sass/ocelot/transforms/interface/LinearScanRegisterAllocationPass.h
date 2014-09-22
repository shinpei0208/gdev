/*! \file LinearScanRegisterAllocationPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Monday December 21, 2009
	\brief The header file for the LinearScanRegisterAllocationPass class.
	*/

#ifndef LINEAR_SCAN_REGISTER_ALLOCATION_PASS_H_INCLUDED
#define LINEAR_SCAN_REGISTER_ALLOCATION_PASS_H_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/interface/DataflowGraph.h>

#include <ocelot/transforms/interface/Pass.h>
#include <ocelot/transforms/interface/MemoryArray.h>
#include <ocelot/transforms/interface/SpillPolicy.h>

// Standard Library Includes
#include <map>

namespace transforms
{

class SpillPolicy;

/*! \brief Implements the linear scan register allocation algorithm */
class LinearScanRegisterAllocationPass: public KernelPass
{
public:
	typedef SpillPolicy::RegisterId RegisterId;
	typedef SpillPolicy::CoalescedRegisterVector CoalescedRegisterVector;
	typedef SpillPolicy::CoalescedRegisterMap CoalescedRegisterMap;
	typedef SpillPolicy::RegisterMap RegisterMap;
	typedef SpillPolicy::RegisterSet RegisterSet;
	
	typedef analysis::DataflowGraph::InstructionVector::iterator
		instruction_iterator;
	
	/*! \brief A set of application points */
	typedef std::set<Interval::Point> PointSet;
	/*! \brief A sorted map, by interval end, of sets of registers */
	typedef std::map<Interval::Point, RegisterSet> PointRegistersMap;
	/*! \brief A sorted map, by interval end, of sets of registers */
	typedef std::map<Interval::Point,
		instruction_iterator> PointInstructionMap;
	/*! \brief A sorted map, by interval end, of sets of registers */
	typedef std::map<Interval::Point,
		analysis::DataflowGraph::iterator> PointBlockMap;
	/*! \brief List of physical registers */
	typedef std::list<RegisterId> RegisterList;

	/*! \brief Choose data type for SSA variables with different data
		types, coalesced into a single variable */
	const ir::PTXOperand::DataType selectType(
		const ir::PTXOperand::DataType &a,
		const ir::PTXOperand::DataType &b) const;

	/*! \brief Constructor on the number of registers to allocate */
	LinearScanRegisterAllocationPass(unsigned regs = 8);
	
	LinearScanRegisterAllocationPass(unsigned regs,
		const Analysis::StringVector& analysis, unsigned reserved);
	
	/*! \brief Initialize the pass using a specific module */
	virtual void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	virtual void runOnKernel(ir::IRKernel& k);
	/*! \brief Finalize the pass */
	void finalize();

void setRegisterCount(unsigned regs);

protected:
	/*! \brief Get an up to date data flow graph */
	analysis::DataflowGraph& _dfg();
	/*! \brief The kernel being operated on */
	ir::IRKernel* _kernel;
	/*! \brief The total set of coalesced registers */
	CoalescedRegisterVector _coalesced;
	/*! \brief The map from SSA to coalesced registers */
	CoalescedRegisterMap _ssa;
	/*! \brief This is a map of all written variables on each point */
	PointRegistersMap _writes;
	/*! \brief This is a map of all instructions sorted by point */
	PointInstructionMap _instructions;
	/*! \brief This is a map that associate block to points */
	PointBlockMap _blocks;
	/*! \brief Program memory stack, that must
		keep alignment by data size */
	CoalescedArray _memoryStack;
	/*! \brief Clears all variables for each kernel pass */
	virtual void _clear();
	/*! \brief The first pass coalesces SSA registers */
	virtual void _coalesce();
	/*! \brief The final pass inserts the spill code */
	virtual void _spill();
	/*! \brief Allocate stack space for the spilled registers */
	virtual void _extendStack();
	SpillPolicy _spillPolicy;
private:
	/*! \brief The second pass computes the live intervals */
	void _computeIntervals();
	/*! \brief The third pass performs register allocation */
	void _allocate();
	/*! \brief Treat alive in points */
	void _treatAliveInPoint(RegisterList &available,
		const Interval::Point pointNum);
	/*! \brief Treat conventional points */
	void _treatPoint(RegisterList &available,
		const Interval::Point intervalNum);
	/*! \brief Create correct coalesced register type */
	virtual void _addCoalesced(const RegisterId id,
		const analysis::DataflowGraph::Type type);
	/*! \brief Sequence of basic blocks stored in program order */
	analysis::DataflowGraph::BlockPointerVector _sequence;
	/*! \brief This is the total number of registers */
	RegisterId _registers;
	/*! \brief This is the set of all positions that are blocks aliveIn */
	PointSet _aliveIns;
	/*! \brief Maps which variables are on register at the beginning
		of each point of the program */
	PointRegistersMap _onRegisters;
};

}

#endif

