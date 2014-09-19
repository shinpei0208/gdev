/*! \file AffineRegister.h
	\author Diogo Nunes Sampaio <dnsampaio@gmail.com>
	\date April 5, 2012
	\brief The header file for the DivergenceRegister class.
*/

#ifndef DIVERGENCE_REGISTER_H_
#define DIVERGENCE_REGISTER_H_

// Ocelot Includes
#include <ocelot/transforms/interface/CoalescedRegister.h>
#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/transforms/interface/MemoryArray.h>

// Standard Library Includes
#include <string>

#define DIVERGENCE_REGISTER_PROFILE_H_ 1

#if DIVERGENCE_REGISTER_PROFILE_H_

namespace divergenceProfiler
{

void resetSpillData();
void printSpillResults(const std::string kernelName);

}
#endif

namespace transforms
{

class DivergenceRegister : public CoalescedRegister
{
public:
	static bool Constant;
	static bool Divergent;
	static analysis::DataflowGraph::RegisterId warpPosition;

private:
	typedef CoalescedRegister::InstructionList InstructionList;
	/*! \brief Tells that variable is divergent */
	bool divergent() const;

	/*! \brief The combined divergence state of the variable */
	bool _state;
	/*! \brief The shared array for spills */
	MemoryArray *_shared;

public:
	/*! \brief Class constructor */
	DivergenceRegister(const RegisterId ireg,
		const Type itype, MemoryArray* local,
		MemoryArray* shared, const bool s = Constant);
	/*! \brief Combine divergence state of of the variable and one
		that is being coalesced to */
	void combineState(const bool state);
	/*! \brief Generate loads instructions for the variable */
	virtual void load(DataflowGraph & dfg, InstructionList &il,
		RegisterId &dreg);
	/*! \brief Generate store instructions for the variable */
	virtual void store(DataflowGraph & dfg, InstructionList &il,
		RegisterId &sreg);
	/*! \brief Spill coalesced variable */
	virtual void spill();
	/*! \brief Tells how many additional registers are required,
		at most, on a load or store process of the variable */
	const bool state() const { return _state;};
};

}

#endif

