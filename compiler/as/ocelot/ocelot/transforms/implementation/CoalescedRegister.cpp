/*
* CoalescedRegister.cpp
*
*	Created on: Oct 21, 2011
*			Author: undead
*/

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#include <ocelot/transforms/interface/CoalescedRegister.h>

namespace transforms
{

CoalescedRegister::Type CoalescedRegister::predType =
	ir::PTXOperand::DataType::u16;

unsigned Interval::size() const
{
	assertM(begin < end, "Variable life does not begins before it ends");
	return (end - begin);
}


CoalescedRegister::CoalescedRegister(RegisterId ireg, Type itype, MemoryArray* array)
: _reg(ireg), _spilled(false), _size(1), _type(itype), _mem(array)
{
	if(ir::PTXOperand::bytes(_type) == 8) _size = 2;
}

CoalescedRegister::~CoalescedRegister()
{

}

CoalescedRegister::RegisterId CoalescedRegister::reg() const
{
	return _reg;
}

CoalescedRegister::Type CoalescedRegister::type() const
{
	return _type;
}

bool CoalescedRegister::spilled() const
{
	return _spilled;
}

uint16_t CoalescedRegister::size() const
{
	return _size;
}

bool CoalescedRegister::predicate() const
{
	return (_type == ir::PTXOperand::DataType::pred);
}

bool CoalescedRegister::isAllocated() const
{
	return allocated.size() == _size;
}

unsigned int CoalescedRegister::readDistance(const Interval::Point &point) const
{
	return distance(point, RW::READ);
}

unsigned int CoalescedRegister::writeDistance(const Interval::Point &point) const
{
	return distance(point, RW::WRITE);
}

unsigned int CoalescedRegister::distance(const Interval::Point &point,
	const RW type) const
{
	UseMap::const_iterator pos = rw.find(point);

	if(pos == rw.end()) pos = rw.begin();

	while(pos != rw.end() && ((pos->first <= point) || !(pos->second & type)))
	{
		pos++;
	}

	if(pos == rw.end()) return (unsigned int)(-1);

	return pos->first - point;
}

void CoalescedRegister::spill()
{
	assertM(!_spilled, "Can't spill a already spilled variable");
	_spilled = true;
	if(_type == ir::PTXOperand::DataType::pred)
	{
		_mem->insertVar(_reg, predType);
	}
	else
	{
		_mem->insertVar(_reg, _type);
	}
}

void CoalescedRegister::load(DataflowGraph & dfg,
	InstructionList &il, RegisterId &dreg)
{
	if(!predicate())
	{
		ir::PTXInstruction ld( ir::PTXInstruction::Opcode::Ld );
		ld.type = _type;
		ld.addressSpace = _mem->addressSpace();
		ld.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Address, _type,
			_mem->name(), _mem->getVarOffset(_reg));
		ld.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
			_type, dreg);
		report("ld:		" << ld.toString());
		il.push_back(ld);
		return;
	}

	RegisterId identifier = dfg.newRegister();

	ir::PTXInstruction ld(ir::PTXInstruction::Opcode::Ld);
	ld.type = predType;
	ld.addressSpace = _mem->addressSpace();
	ld.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Address,
		predType, _mem->name(), _mem->getVarOffset(_reg));
	ld.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, identifier);
	il.push_back(ld);
	report("ld:		" << ld.toString() );

	ir::PTXInstruction setp(ir::PTXInstruction::Opcode::SetP);
	setp.type = predType;
	setp.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, identifier);
	setp.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		_type, dreg);
	setp.b = ir::PTXOperand(1, predType);
	setp.comparisonOperator = ir::PTXInstruction::CmpOp::Eq;
	report("setp:		" << setp.toString() );
	il.push_back(setp);
}

void CoalescedRegister::store(DataflowGraph & dfg,
	InstructionList &il, RegisterId &sreg)
{
	if(!predicate())
	{
		ir::PTXInstruction st(ir::PTXInstruction::Opcode::St);
		st.type = _type;
		st.addressSpace = _mem->addressSpace();
		st.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Address, _type,
			_mem->name(), _mem->getVarOffset(_reg));
		st.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
			_type, sreg);
		report("st:		" << st.toString());
		il.push_back(st);
		return;
	}
	
	RegisterId identifier = dfg.newRegister();

	ir::PTXInstruction selp(ir::PTXInstruction::Opcode::SelP);
	selp.type = predType;
	selp.a = ir::PTXOperand(1, predType);
	selp.b = ir::PTXOperand(0, predType);
	selp.c = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		_type, sreg);
	selp.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, identifier);
	report("selp:		" << selp.toString());
	il.push_back(selp);

	ir::PTXInstruction st(ir::PTXInstruction::Opcode::St);
	st.type = predType;
	st.addressSpace = _mem->addressSpace();
	st.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Address, predType,
		_mem->name(), _mem->getVarOffset(_reg));
	st.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Register, predType, 
		identifier);
	report("st:		" << st.toString());
	il.push_back(st);

}

unsigned int CoalescedRegister::averageDistance() const
{
	if(rw.size() < 2) return (unsigned int)(-1);

	return ( (std::prev(rw.end())->first - rw.begin()->first) / rw.size());
}

} /* namespace transforms */

