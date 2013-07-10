/*! \file AffineRegister.h
 \author Diogo Nunes Sampaio <dnsampaio@gmail.com>
 \date Wednesday February 15, 2012
 \brief The file for the AffineRegister class.
 */

#ifndef AFFINE_REGISTER_CPP_
#define AFFINE_REGISTER_CPP_

// Ocelot Includes
#include <ocelot/transforms/interface/AffineRegister.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/SystemCompatibility.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif
#define REPORT_BASE 1

#if AFFINE_REGISTER_PROFILE_H_
#include <fstream>

namespace affineProfiler
{

unsigned divSpills  = 0;
unsigned affSpills  = 0;
unsigned caffSpills = 0;
unsigned unifSpills = 0;
unsigned cstSpills  = 0;
unsigned divLoads   = 0;
unsigned affLoads   = 0;
unsigned caffLoads  = 0;
unsigned unifLoads  = 0;
unsigned cstLoads   = 0;
unsigned divStores  = 0;
unsigned affStores  = 0;
unsigned caffStores = 0;
unsigned unifStores = 0;
unsigned cstStores  = 0;

void resetSpillData()
{
	divSpills = affSpills = caffSpills = unifSpills = cstSpills = 0;
	divLoads  = affLoads  = caffLoads  = unifLoads  = cstLoads  = 0;
	divStores = affStores = caffStores = unifStores = cstStores = 0;
}

void printSpillResults(const std::string kernelName)
{
	string k = hydrazine::demangleCXXString(kernelName);

	//Remove parameter from function name
	k = k.substr(0, k.find("("));
	
	//Remove data type from templated kernel
	if(k.find('<') != string::npos)
	{
		k = k.substr(0, k.find("<"));
	}
	
	//Remove function namespace from templated kernel
	if(k.find(':') != string::npos)
	{
		k.replace(0, 1 + k.find_last_of(':'), "");
	}
	
	//Remove function type from templated kernel
	if(k.find(' ') != string::npos)
	{
		k.replace(0, 1 + k.find_last_of(' '), "");
	}

	std::ofstream out(k + ".AffSpill.csv");
	if(!out.is_open()) return;

	out << "div;aff;C.aff;unif;cst;divLD;affLD;caffLD;uniLD;CstLD;"
		"divST;affST;C.affST;uniST;ConstST"
		<< std::endl << divSpills << ';' << affSpills << ';' << caffSpills
		<< ';' << unifSpills << ';' << cstSpills << ';'
		<< divLoads << ';' << affLoads << ';' << caffLoads << ';' << unifLoads
		<< ';' << cstLoads << ';' << divStores << ';'
		<< affStores << ';' << caffStores << ';' << unifStores << ';'
		<< cstStores;

	out.close();
}

}

#endif

namespace transforms
{

AffineRegister::RegisterId AffineRegister::warpPosition = (unsigned) (-1);

std::map<ir::PTXOperand::DataType, AffineRegister::RegisterId>
	AffineRegister::tempRegisters;

bool AffineRegister::bottomBase() const
{
	return _state.base == analysis::ConstantAbstractState::bottom;
}

bool AffineRegister::baseZero() const
{
	return _state.base == analysis::ConstantAbstractState::zero;
}

bool AffineRegister::strideOne() const
{
	return _state.stride[0] == analysis::ConstantAbstractState::one;
}

bool AffineRegister::doNotRequireMemory() const
{
	return _state.known();
}

bool AffineRegister::requireSharedMemory() const
{
	return (!(requireLocalMemory() || doNotRequireMemory()));
}

bool AffineRegister::requireLocalMemory() const
{
	return (_state.undefined() || _state.divergent() || _state.hardAffine());
}

AffineRegister::AffineRegister(RegisterId ireg, Type itype,
	MemoryArray* localArray, MemoryArray* affineStack,
	const analysis::AffineAbstractState state)
: CoalescedRegister(ireg, itype, localArray), _shared(affineStack), _regs(0)
{

}

void AffineRegister::combineState(const analysis::AffineAbstractState state)
{
	_state &= state;
}

void AffineRegister::load(DataflowGraph & dfg, InstructionList &il,
	RegisterId &dreg)
{
	report("Load coalesced variable " << _reg << ", SSA variable " << dreg);
	
	if(requireLocalMemory())
	{
#if AFFINE_REGISTER_PROFILE_H_
		affineProfiler::divLoads++;
#endif
		report("\tGoes into local memory");
		CoalescedRegister::load(dfg, il, dreg);
		return;
	}

	if(predicate())
	{
		report("\tIs a affine or constant predicate");
		loadPred(dfg, il, dreg);
		return;
	}

	if(doNotRequireMemory())
	{
		report("\tIs a affine or constant variable with known indexes");
		recomputeKnownValue(il, dreg);
		return;
	}
	
	report("\tKnown stride, unknown base");

	if(_state.isUniform())
	{
#if AFFINE_REGISTER_PROFILE_H_
		affineProfiler::unifLoads++;
#endif

		/* 0 + 0 + B */
		std::stringstream a;
		a << "%r" << warpPosition;
		if(_shared->getVarOffset(_reg) != 0)
		{
			a << " + " << _shared->getVarOffset(_reg);
		}
		ir::PTXInstruction load(ir::PTXInstruction::Ld);
		load.addressSpace = _shared->addressSpace();
		load.d = ir::PTXOperand(ir::PTXOperand::Register, _type, dreg);
		load.a = ir::PTXOperand(a.str());

		load.type = _type;
		il.push_back(load);
		return;
	}
#if AFFINE_REGISTER_PROFILE_H_
	affineProfiler::affLoads++;
#endif
	/* 0 + C + B */
	std::stringstream a;
	a << "%r" << warpPosition;
	if(_shared->getVarOffset(_reg) != 0)
		a << " + " << _shared->getVarOffset(_reg);

	ir::PTXInstruction load(ir::PTXInstruction::Ld);
	load.addressSpace = _shared->addressSpace();
	load.d = ir::PTXOperand(ir::PTXOperand::Register, _type, dreg);
	load.a = ir::PTXOperand(a.str());
	load.type = _type;
	il.push_back(load);

	report("\t\tIs not constant");
	ir::PTXInstruction::Opcode mvOpc = ir::PTXInstruction::Opcode::Mov;
	
	if(_type != ir::PTXOperand::DataType::u32)
	{
		mvOpc = ir::PTXInstruction::Opcode::Cvt;
	}
	
	if(tempRegisters.find(_type) == tempRegisters.end())
	{
		tempRegisters[_type] = dfg.newRegister();
	}
	
	ir::PTXInstruction mv(mvOpc);
	mv.type = _type;
	mv.a = ir::PTXOperand(ir::PTXOperand::SpecialRegister::tid,
		ir::PTXOperand::VectorIndex::ix);
	mv.d = ir::PTXOperand(ir::PTXOperand::Register, _type,
		tempRegisters[_type]);
	il.push_back(mv);

	if(strideOne())
	{
		report("\t\tHas stride 1");
		ir::PTXInstruction add(ir::PTXInstruction::Opcode::Add);
		add.type = _type;
		add.a = mv.d;
		add.b = load.d;
		add.d = ir::PTXOperand(ir::PTXOperand::Register, _type, dreg);
		il.push_back(add);
	}
	else
	{
		report("\t\tHas stride != 1" << _state.stride[0].value);
		ir::PTXInstruction mad(ir::PTXInstruction::Opcode::Mad);
		mad.a = mv.d;
		mad.b = ir::PTXOperand(_state.stride[0].value, _type);
		mad.c = load.d;
		mad.d = ir::PTXOperand(ir::PTXOperand::Register, _type, dreg);
		mad.type = _type;
		mad.modifier = ir::PTXInstruction::Modifier::lo;
		il.push_back(mad);
	}
	return;
}

void AffineRegister::loadPred(DataflowGraph &dfg, InstructionList &il,
	RegisterId &dreg)
{
	if(tempRegisters.find(predType) == tempRegisters.end())
	{
		tempRegisters[predType] = dfg.newRegister();
	}
	
#if AFFINE_REGISTER_PROFILE_H_
	affineProfiler::affLoads++;
#endif
	std::stringstream s;
	s << "%r" << warpPosition;
	if(_shared->getVarOffset(_reg) != 0)
	{
		s << " + " << _shared->getVarOffset(_reg);
	}
	
	ir::PTXInstruction ld(ir::PTXInstruction::Opcode::Ld);
	ld.type = predType;
	ld.addressSpace = _shared->addressSpace();
	ld.a = ir::PTXOperand(s.str());
	ld.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, tempRegisters[predType]);
	il.push_back(ld);
	report("ld:		" << ld.toString());

	ir::PTXInstruction setp(ir::PTXInstruction::Opcode::SetP);
	setp.type = predType;
	setp.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, tempRegisters[predType]);
	setp.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register, _type, dreg);
	setp.b = ir::PTXOperand(1, predType);
	setp.comparisonOperator = ir::PTXInstruction::CmpOp::Eq;
	report("setp:		" << setp.toString());
	il.push_back(setp);
}

void AffineRegister::storePred(DataflowGraph & dfg,
	InstructionList &il, RegisterId &sreg)
{
	if(tempRegisters.find(predType) == tempRegisters.end())
	{
		tempRegisters[predType] = dfg.newRegister();
	}
	
#if AFFINE_REGISTER_PROFILE_H_
	affineProfiler::affStores++;
#endif

	std::stringstream s;
	s << "%r" << warpPosition;
	if(_shared->getVarOffset(_reg) != 0)
	{
		s << " + " << _shared->getVarOffset(_reg);
	}
	
	ir::PTXInstruction selp(ir::PTXInstruction::Opcode::SelP);
	selp.type = predType;
	selp.a = ir::PTXOperand(1, predType);
	selp.b = ir::PTXOperand(0, predType);
	selp.c = ir::PTXOperand(ir::PTXOperand::AddressMode::Register, _type, sreg);
	selp.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, tempRegisters[predType]);
	report("selp:		" << selp.toString());
	il.push_back(selp);

	ir::PTXInstruction st(ir::PTXInstruction::Opcode::St);
	st.type = predType;
	st.addressSpace = _shared->addressSpace();
	st.d = ir::PTXOperand(ir::PTXOperand::AddressMode::Address, predType,
		_shared->name(), _shared->getVarOffset(_reg));
	st.a = ir::PTXOperand(ir::PTXOperand::AddressMode::Register,
		predType, tempRegisters[predType]);
	report("st:		" << st.toString());
	il.push_back(st);
}

void AffineRegister::recomputeKnownValue(InstructionList &il, RegisterId &dreg)
{
	/* 0 + C + C */
	report("Recomputing known value: " << _state)
	if(!_state.affine())
	{
#if AFFINE_REGISTER_PROFILE_H_
		affineProfiler::cstLoads++;
#endif
		report("\tIs constant");
		/* 0 + 0 + C*/
		ir::PTXInstruction mv(ir::PTXInstruction::Opcode::Mov);
		mv.type = _type;
		mv.d = ir::PTXOperand(ir::PTXOperand::Register, _type, dreg);
		mv.a = ir::PTXOperand(_state.base.value, _type);
		il.push_back(mv);
		return;
	}
	
#if AFFINE_REGISTER_PROFILE_H_
	affineProfiler::caffLoads++;
#endif

	/* 0 + C + K, First we load the constant C, or stride */
	report("\tIs not constant");

	ir::PTXInstruction::Opcode mvOpc = ir::PTXInstruction::Opcode::Mov;
	if(_type != ir::PTXOperand::DataType::u32)
	{
		mvOpc = ir::PTXInstruction::Opcode::Cvt;
	}
	
	ir::PTXInstruction mv(mvOpc);
	mv.d = ir::PTXOperand(ir::PTXOperand::Register, _type, dreg);
	mv.a = ir::PTXOperand(ir::PTXOperand::SpecialRegister::tid,
		ir::PTXOperand::VectorIndex::ix);
	mv.type = _type;
	il.push_back(mv);

	if(strideOne())
	{
		if(!baseZero())
		{
			/* 0 + 1 + K, Just add K to the loaded value */
			report("\t\t\tHas stride = 1 and base != 0");
			ir::PTXInstruction add(ir::PTXInstruction::Opcode::Add);
			add.d = mv.d;
			add.a = mv.d;
			add.b = ir::PTXOperand(_state.stride[0].value, _type);
			add.type = _type;
			il.push_back(add);
		}
		return;
	}

	if(baseZero())
	{
		report("\t\t\tHas stride != 1 and base = 0");
		ir::PTXInstruction mul(ir::PTXInstruction::Opcode::Mul);
		mul.d = mv.d;
		mul.a = mv.d;
		mul.b = ir::PTXOperand(_state.stride[0].value, _type);
		mul.type = _type;
		mul.modifier = ir::PTXInstruction::Modifier::lo;
		il.push_back(mul);
		return;
	}

	report("\t\t\tHas stride != 1 and base != 0");
	/* 0 + C != 1 + C */
	ir::PTXInstruction mad(ir::PTXInstruction::Opcode::Mad);
	mad.d = mv.d;
	mad.a = mv.d;
	mad.b = ir::PTXOperand(_state.stride[0].value, _type);
	mad.c = ir::PTXOperand(_state.base.value, _type);
	mad.type = _type;
	mad.modifier = ir::PTXInstruction::Modifier::lo;
	il.push_back(mad);
}

void AffineRegister::store(DataflowGraph &dfg,
	InstructionList &il, RegisterId &sreg)
{
	report("Affine store for variable " << sreg
		<< ", coalesced to " << _reg << ", state: " << _state);
	
	if(requireLocalMemory())
	{
#if AFFINE_REGISTER_PROFILE_H_
		affineProfiler::divStores++;
#endif
		CoalescedRegister::store(dfg, il, sreg);
		return;
	}

	if(predicate())
	{
		storePred(dfg, il, sreg);
		return;
	}

	if(doNotRequireMemory())
	{
#if AFFINE_REGISTER_PROFILE_H_
		if(_state.affine())
		{
			affineProfiler::caffStores++;
		}
		else
		{
			affineProfiler::cstStores++;
		}
#endif
		report("No store required, needs to recreate data");
		return;
	}

	report("Has bottom base");

	RegisterId store = sreg;
	
	if(_state.isUniform())
	{
#if AFFINE_REGISTER_PROFILE_H_
		affineProfiler::unifStores++;
#endif
		std::stringstream s;
		s << "%r" << warpPosition;
		if(_shared->getVarOffset(_reg) != 0)
			s << " + " << _shared->getVarOffset(_reg);

		ir::PTXInstruction st(ir::PTXInstruction::St);
		st.d = ir::PTXOperand(s.str());
		st.a = ir::PTXOperand(ir::PTXOperand::Register, _type, store);
		st.addressSpace = _shared->addressSpace();
		st.type = _type;
		il.push_back(st);
		return;
	}
	
#if AFFINE_REGISTER_PROFILE_H_
	affineProfiler::affStores++;
#endif

	report("Has stride != 0");
	ir::PTXInstruction::Opcode mvOpc = ir::PTXInstruction::Opcode::Mov;
	
	if(_type != ir::PTXOperand::DataType::u32)
	{
		mvOpc = ir::PTXInstruction::Opcode::Cvt;
	}
	
	if(tempRegisters.find(_type) == tempRegisters.end())
	{
		tempRegisters[_type] = dfg.newRegister();
	}
	
	store = tempRegisters[_type];

	ir::PTXInstruction mv(mvOpc);
	mv.d = ir::PTXOperand(ir::PTXOperand::Register, _type, store);
	mv.a = ir::PTXOperand(ir::PTXOperand::SpecialRegister::tid,
		ir::PTXOperand::VectorIndex::ix);
	mv.type = _type;
	il.push_back(mv);

	if(strideOne())
	{
		report("Has stride == 1");
		ir::PTXInstruction sub(ir::PTXInstruction::Opcode::Sub);
		sub.d = mv.d;
		sub.a = ir::PTXOperand(ir::PTXOperand::Register, _type, sreg);
		sub.b = mv.d;
		sub.type = _type;
		il.push_back(sub);
	}
	else
	{
		report("Has stride != 1");
		ir::PTXInstruction mad(ir::PTXInstruction::Opcode::Mad);
		mad.d = mv.d;
		mad.a = mv.d;
		mad.b = ir::PTXOperand(-_state.stride[0].value, _type);
		mad.c = ir::PTXOperand(ir::PTXOperand::Register, _type, sreg);
		mad.type = _type;
		mad.modifier = ir::PTXInstruction::Modifier::lo;
		il.push_back(mad);
	}

	std::stringstream s;
	s << "%r" << warpPosition;

	if(_shared->getVarOffset(_reg) != 0)
	{
		s << " + " << _shared->getVarOffset(_reg);
	}
	
	ir::PTXInstruction st(ir::PTXInstruction::St);
	st.d = ir::PTXOperand(s.str());
	st.a = ir::PTXOperand(ir::PTXOperand::Register, _type, store);
	st.addressSpace = _shared->addressSpace();
	st.type = _type;
	il.push_back(st);
}

void AffineRegister::spill()
{
	if(_spilled) return;

	if(requireLocalMemory())
	{
#if AFFINE_REGISTER_PROFILE_H_
		if(!_spilled) affineProfiler::divSpills++;
#endif

		CoalescedRegister::spill();
		return;
	}
	
	_spilled = true;
	
	if(doNotRequireMemory())
	{
#if AFFINE_REGISTER_PROFILE_H_
		if(_state.affine())
		{
			affineProfiler::caffSpills++;
		}
		else
		{
			affineProfiler::cstSpills++;
		}
#endif
		return;
	}

	if(_type == ir::PTXOperand::DataType::pred)
	{
#if AFFINE_REGISTER_PROFILE_H_
		affineProfiler::affSpills++;
#endif
		_shared->insertVar(_reg, predType);
	}
	else
	{
		_shared->insertVar(_reg, _type);
#if AFFINE_REGISTER_PROFILE_H_
		if(_state.affine())
			affineProfiler::affSpills++;
		else
			affineProfiler::unifSpills++;
#endif
	}
}

unsigned AffineRegister::additionalRegs() const
{
	return _regs;
}

}

#endif

