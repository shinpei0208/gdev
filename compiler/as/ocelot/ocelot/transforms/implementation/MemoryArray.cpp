/*
*  MemoryArray.cpp
*
*	Created on: Oct 28, 2011
*			Author: undead
*/

#include <ocelot/transforms/interface/MemoryArray.h>

#include <hydrazine/interface/debug.h>

#include <iostream>
#include <string>
#include <sstream>

namespace transforms
{

MemoryArray::MemoryArray(const std::string & arrayname,
	const MemoryDirective dir, const StackAddressSpace ads)
: _name(arrayname), _addressSpace(ads), _dir(dir), _stackSize(0),
	_minVarSize(-1), _maxVarSize(0), _hasUnsigned(false), _hasInt(false),
	_hasFloat(false), _isBin(false){
}

MemoryArray::~MemoryArray(){
	clear();
};

void MemoryArray::declaration(ir::Kernel::LocalMap &map, unsigned factor){
	if (_stackSize == 0) return;
	ir::PTXStatement stack(_dir);

	stack.type = type();
	stack.name = _name;
	stack.alignment = _maxVarSize;
	stack.array.stride.push_back(bytes(factor != 1) * factor);

	map.insert(std::make_pair(_name, ir::Local(stack)));
}

const std::string& MemoryArray::name() const{
	return _name;
}

MemoryArray::StackAddressSpace MemoryArray::addressSpace() const{
	return _addressSpace;
}

MemoryArray::Type MemoryArray::type() const
{
	unsigned deslocation = 0;
	unsigned minSize = _minVarSize;
	while(minSize > 1){
		minSize >>= 1;
		deslocation++;
	}
	/* If the stack has different types (float/int) or has bin,
	 * then it must be a bin */
	if(_isBin){
		return (Type)((unsigned)(Type::b8) + deslocation);
	} else if(_hasFloat){ /*Has only floats */
		assertM(deslocation > 0, "Has only floats, but min size = 8");
		return (Type)((unsigned)(Type::f16) + deslocation - 1);
		/* If there is signed and unsigned, can use unsigned */
	} else if(_hasUnsigned){
		return (Type)((unsigned)(Type::u8) + deslocation);
	}
	return (Type)((unsigned)(Type::s8) + deslocation);
}

unsigned MemoryArray::stackAlignment() const{
	return _maxVarSize;
}

unsigned MemoryArray::bytes(bool tiled){
	if( (tiled) && ((_stackSize % _maxVarSize) != 0) ){
		_stackSize += _maxVarSize - (_stackSize % _maxVarSize);
	}
	return _stackSize;
}

unsigned MemoryArray::elements() const{
	return _declared.size();
}

unsigned MemoryArray::physicalElements() const{
	if(elements() == 0)
		return 0;
	assertM(((_stackSize % _minVarSize) == 0), "Not divisible stack size by minimal variable size");
	return _stackSize / _minVarSize;
}

bool MemoryArray::hasVar(const RegisterId reg) const{
	return _declared.find(reg) != _declared.end();
}

void MemoryArray::clear(){
	_declared.clear();
	_hasFloat = false;
	_hasInt = false;
	_hasUnsigned = false;
	_isBin = false;
	_maxVarSize = 0;
	_minVarSize = (unsigned int)(-1);
	_stackSize = 0;
}

bool MemoryArray::insertVar(const RegisterId reg, const Type type)
{
	assertM(type != Type::pred, "Can't put a predicate to memory, reg id:" << reg);

	if(hasVar(reg)){
		assertM(type == _declared.find(reg)->second, "Tried to insert different" <<
				" types for same identifier, id:" << reg << ", new type:" <<
				ir::PTXOperand::toString(type) << ", old type: "
				<< ir::PTXOperand::toString(_declared.find(reg)->second));
		return false;
	}

	_minVarSize = std::min(_minVarSize, ir::PTXOperand::bytes(type));
	_maxVarSize = std::max(_maxVarSize, ir::PTXOperand::bytes(type));
	_isBin |= (type >= Type::b8);
	if(!_isBin){
		_hasFloat |= ir::PTXOperand::isFloat(type);
		_hasInt |= ir::PTXOperand::isInt(type);
		_hasUnsigned |= (!ir::PTXOperand::isSigned(type));
		_isBin = (_hasFloat & _hasInt);
	}
	return true;
}

InOrderArray::InOrderArray(const std::string & arrayname,
	const MemoryDirective dir, const StackAddressSpace ads)
: MemoryArray(arrayname, dir, ads)
{

}

bool InOrderArray::insertVar(const RegisterId reg, const Type type)
{
	if(!MemoryArray::insertVar(reg, type)) return false;

	_declared[reg] = type;
	unsigned varSize = ir::PTXOperand::bytes(type);
	unsigned offset = (_stackSize % varSize);

	if(offset != 0) offset = varSize - offset;

	offset    += _stackSize;
	_mem[reg]  = offset;
	_stackSize = offset + varSize;

	return true;
}

int InOrderArray::getVarOffset(const RegisterId reg)
{
	IoMemOffset::const_iterator off = _mem.find(reg);
	assertM((off != _mem.end()), "Variable " << reg
		<< " not declared here: " << _name);

	return off->second;
}

void	InOrderArray::clear()
{
	MemoryArray::clear();
	_mem.clear();
}

CoalescedArray::CoalescedArray(const std::string & arrayname,
	const MemoryDirective dir, const StackAddressSpace ads)
: MemoryArray(arrayname, dir, ads), _canInsert(true)
{

}

bool CoalescedArray::insertVar(const RegisterId reg, const Type type)
{
	assertM(_canInsert, "Can't insert new values after getting a offset");
	if(!MemoryArray::insertVar(reg, type)) return false;
	
	_declared[reg] = type;
	unsigned varSize = ir::PTXOperand::bytes(type);
	
	if(_mem.find(varSize) == _mem.end())
	{
		std::set<RegisterId> tmp;
		_mem[varSize] = tmp;
	}
	
	_mem[varSize].insert(reg);
	_stackSize += varSize;
	return true;
}

int CoalescedArray::getVarOffset(const RegisterId reg)
{
	_canInsert = false;
	auto t = _declared.find(reg);
	
	if(t == _declared.end())
	{
		assertM(false, "Variable " << reg << " not declared here");
		return -1;
	}

	unsigned offset  = 0;
	unsigned varSize = ir::PTXOperand::bytes(t->second);
	
	CoMemOffset::const_iterator size = _mem.begin();
	CoMemOffset::const_iterator sizeEnd = _mem.end();
	
	for(; size != sizeEnd; size++)
	{
		if(size->first == varSize)
		{
			std::set<RegisterId>::iterator r = size->second.find(reg);

			if(r == size->second.end()) return -1;

			offset += varSize*(std::distance(size->second.begin(), r));
		}
		else if (size->first > varSize)
		{
			offset += size->first * size->second.size();
		}
	}
	return offset;
}

void CoalescedArray::clear()
{
	MemoryArray::clear();
	_mem.clear();
	_canInsert = true;
}

} /* namespace transforms */

