/*
* CoalescedRegister.h
*
*	Created on: Oct 21, 2011
*			Author: undead
*/

#ifndef COALESCEDREGISTER_H_
#define COALESCEDREGISTER_H_

// Ocelot Includes
#include <ocelot/ir/interface/PTXStatement.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/transforms/interface/MemoryArray.h>

// Standard Library Includes
#include <stack>

namespace transforms
{

/*! \brief The range of instructions for which a SSA variable is alive */
class Interval
{
public:
	typedef unsigned Point;
	Interval() :
			begin((unsigned int) (-1)), end(0){
	}
	;
	/*! \brief The first instruction in the interval */
	Point begin;
	/*! \brief One past the last instruction in the interval */
	Point end;
	/*! \brief Interval size */
	unsigned size() const;
};

/*! \brief A coalesced register represents aliased SSA registers */
class CoalescedRegister
{
friend class LinearScanRegisterAllocationPass;

public:
	typedef analysis::DataflowGraph DataflowGraph;
	typedef DataflowGraph::RegisterId RegisterId;
	typedef ir::PTXOperand::DataType Type;
	typedef std::list<ir::PTXInstruction> InstructionList;

	enum RW
	{
		READ = 1,
		WRITE = 2
	};

	typedef std::map<Interval::Point, RW> UseMap;

	/*! \brief To which data type predicates are converted to be stored on memory */
	static Type predType;
	unsigned int readsCount;
	unsigned int writesCount;

protected:
	/*! \brief The unique id of the register */
	RegisterId _reg;
	/*! \brief Has the register been spilled */
	bool _spilled;
	/*! \brief The number of physical register required */
	unsigned short _size;
	/*! \brief The type of the register */
	Type _type;
	/*! \brief Reference to memory stack where the register might be spilled */
	MemoryArray *_mem;
	/*! \brief Read points */
	UseMap rw;
	
public:
	/* \brief The live interval for the register */
	Interval interval;
	/*! \brief The allocated name of the register */
	std::set<RegisterId> allocated;

	CoalescedRegister(RegisterId ireg, Type itype, MemoryArray* array);
	virtual ~CoalescedRegister();

	RegisterId reg() const;

	Type type() const;

	bool spilled() const;

	unsigned short size() const;

	bool predicate() const;

	bool isAllocated() const;

	/*! \brief Tells the distance for the next read of the variable */
	unsigned int readDistance(const Interval::Point &point) const;
	/*! \brief Tells the distance for the next write of the variable */
	unsigned int writeDistance(const Interval::Point &point) const;
	/*! \brief Tells the distance for the next use of the variable */
	unsigned int distance(const Interval::Point &point, const RW type) const;
	/*! \brief Tells the average distance between reads of the variable */
	unsigned int averageDistance() const;

	virtual void spill();
	virtual void load(DataflowGraph & dfg,
		InstructionList &il, RegisterId &dreg);
	virtual void store(DataflowGraph & dfg,
		InstructionList &il, RegisterId &sreg);

};

} /* namespace transforms */

#endif /* COALESCEDREGISTER_H_ */

