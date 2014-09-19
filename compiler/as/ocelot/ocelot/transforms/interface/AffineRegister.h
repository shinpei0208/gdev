/*! \file AffineRegister.h
	\author Diogo Nunes Sampaio <dnsampaio@gmail.com>
	\date Wednesday February 15, 2012
	\brief The header file for the AffineRegister class.
*/

#ifndef AFFINE_REGISTER_H_
#define AFFINE_REGISTER_H_

#define AFFINE_REGISTER_PROFILE_H_ 1

#if AFFINE_REGISTER_PROFILE_H_
// Standard Library Includes
#include <string>
namespace affineProfiler
{
	void resetSpillData();
	void printSpillResults(const std::string kernelName);
}
#endif

// Ocelot Includes
#include <ocelot/transforms/interface/CoalescedRegister.h>
#include <ocelot/analysis/interface/AffineAnalysis.h>
#include <ocelot/transforms/interface/MemoryArray.h>

namespace transforms
{

class AffineRegister : public CoalescedRegister
{
private:
	typedef CoalescedRegister::InstructionList InstructionList;
	/*! \brief Tells that b in V = ax + b is bottom */
	bool bottomBase() const;
	/*! \brief Tells that b in V = ax + b is a constant */
	bool baseZero() const;
	/*! \brief Tells that a in V = ax + b is bottom */
	bool strideOne() const;
	/*! \brief Tells that variable won't require memory
		storage if spilled */
	bool doNotRequireMemory() const;
	/*! \brief Tells that variable will required shared memory
		storage if spilled */
	bool requireSharedMemory() const;
	/*! \brief Tells that variable will required local
		memory storage if spilled */
	bool requireLocalMemory() const;
	/*! \brief Writes the instructions required to do a load of a
		predicate that can be spilled to shared memory */
	void loadPred(DataflowGraph & dfg, InstructionList &il,
		RegisterId &dreg);
	/*! \brief Writes the instructions required to do a load of
		a predicate that can be spilled to shared memory */
	void storePred(DataflowGraph & dfg, InstructionList &il,
		RegisterId &sreg);
	/*! \brief Writes the instructions required to do rematerialize a
		variable that doesn't require memory if spilled */
	void recomputeKnownValue(InstructionList &il, RegisterId &dreg);

	/*! \brief The combined affine state of the variable */
	analysis::AffineAbstractState _state;
	/*! \brief The shared array for spills */
	MemoryArray *_shared;
	/*! \brief Tells how many additional registers are required,
		at most, on a load or store process of the variable */
	unsigned _regs;

public:
	/*! \brief The register that holds the base position of
		shared memory for each warp */
	static RegisterId warpPosition;
	/*! \brief Keep temporary registers id by type, to avoid
		creating too many temporary registers */
	static std::map<ir::PTXOperand::DataType, RegisterId> tempRegisters;
	/*! \brief Class constructor */
	AffineRegister(RegisterId ireg, Type itype, MemoryArray* array,
		MemoryArray* affineStack,
		const analysis::AffineAbstractState state =
		analysis::AffineAbstractState::top);
	/*! \brief Combine affine state of of the variable
		and one that is being coalesced to */
	void combineState(const analysis::AffineAbstractState state);
	/*! \brief Generate loads instructions for the variable */
	virtual void load(DataflowGraph & dfg,
		InstructionList &il, RegisterId &dreg);
	/*! \brief Generate store instructions for the variable */
	virtual void store(DataflowGraph & dfg,
		InstructionList &il, RegisterId &sreg);
	/*! \brief Spill coalesced variable */
	virtual void spill();
	/*! \brief Tells how many additional registers are required, at most,
		on a load or store process of the variable */
	unsigned additionalRegs() const;

	const analysis::AffineAbstractState state() const { return _state; }

};

}

#endif
