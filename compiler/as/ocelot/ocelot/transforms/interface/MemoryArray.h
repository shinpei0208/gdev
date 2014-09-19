/*! \file MemoryAligner.h
	\author Diogo Sampaio<dnsampaio@gmail.com>
	\date 28 Oct 2019
	\brief Keeps information of memory stacks, such as memory for spills
*/

#ifndef MEMORYALIGNER_H_
#define MEMORYALIGNER_H_

#include <string>
#include <stack>
#include <list>
#include <set>
#include <map>

#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/PTXStatement.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/ir/interface/Kernel.h>

namespace transforms
{

class MemoryArray
{
public:
	typedef ir::PTXOperand::AddressMode OperandAddressMode;
	typedef ir::PTXInstruction::AddressSpace StackAddressSpace;
	typedef ir::PTXOperand::DataType Type;
	typedef ir::PTXStatement::Directive MemoryDirective;
	typedef analysis::DataflowGraph::RegisterId RegisterId;
	typedef std::stack<ir::PTXInstruction> Instructions;
	typedef std::stack<ir::PTXStatement> Statements;

	MemoryArray(const std::string & arrayname,
		const MemoryDirective dir = MemoryDirective::Local,
		const StackAddressSpace ads = StackAddressSpace::Local);
	virtual ~MemoryArray();

	void declaration(ir::Kernel::LocalMap &map, const unsigned factor = 1);

	const std::string& name() const;
	StackAddressSpace addressSpace() const;
	Type type() const;
	unsigned stackAlignment() const;
	unsigned bytes(bool tiled = false);
	unsigned elements() const;
	unsigned physicalElements() const;

	bool hasVar(const RegisterId) const;

	void clear();

	virtual int getVarOffset(const RegisterId) = 0;
	
	virtual bool insertVar(const RegisterId, const Type);

protected:
	std::string _name;
	StackAddressSpace _addressSpace;
	MemoryDirective _dir;

	unsigned _stackSize;

	unsigned _minVarSize, _maxVarSize;
	bool _hasUnsigned, _hasInt, _hasFloat, _isBin;

	std::map<RegisterId, Type> _declared;

};

class InOrderArray: public MemoryArray
{
public:
	typedef std::map<RegisterId, unsigned> IoMemOffset;

	InOrderArray(const std::string & arrayname,
				const MemoryDirective dir = MemoryDirective::Local,
				const StackAddressSpace ads = StackAddressSpace::Local);
	bool insertVar(const RegisterId, const Type);
	virtual int getVarOffset(const RegisterId);
	void clear();
private:
	IoMemOffset _mem;
};

class CoalescedArray: public MemoryArray
{
public:
	typedef std::map<unsigned, std::set<RegisterId> > CoMemOffset;

	CoalescedArray(const std::string & arrayname,
		const MemoryDirective dir = MemoryDirective::Local,
		const StackAddressSpace ads = StackAddressSpace::Local);
	bool insertVar(const RegisterId, const Type);
	virtual int getVarOffset(const RegisterId);
	void clear();
private:
	CoMemOffset _mem;
	bool _canInsert;
};

} //	namespace transforms

#endif//	MEMORYALIGNER_H_

