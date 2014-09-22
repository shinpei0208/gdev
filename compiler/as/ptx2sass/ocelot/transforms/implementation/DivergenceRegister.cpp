/*! \file DivergenceRegister.h
	 \author Diogo Nunes Sampaio <dnsampaio@gmail.com>
	 \date Wednesday February 15, 2012
	 \brief The file for the DivergenceRegister class.
 */

// Ocelot Includes
#include <ocelot/transforms/interface/DivergenceRegister.h>

// Hydrazine Includes
#include <hydrazine/interface/SystemCompatibility.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif
#define REPORT_BASE 1

#if DIVERGENCE_REGISTER_PROFILE_H_

#include <fstream>

namespace divergenceProfiler 
{

static unsigned divSpills = 0;
static unsigned cstSpills = 0;
static unsigned divLoads  = 0;
static unsigned cstLoads  = 0;
static unsigned divStores = 0;
static unsigned cstStores = 0;

void resetSpillData()
{
	divSpills = cstSpills = divLoads = cstLoads = divStores = cstStores = 0;
}

void printSpillResults(const std::string kernelName) {

	std::string k = hydrazine::demangleCXXString(kernelName);
	if(k != kernelName) {
		//Remove parameter from function name
		k = k.substr(0, k.find("("));
		//Remove data type from templated kernel
		if(k.find('<') != std::string::npos) {
			k = k.substr(0, k.find("<"));
		}
		
		//Remove function namespace from templated kernel
		if(k.find(':') != std::string::npos) {
			k.replace(0, 1 + k.find_last_of(':'), "");
		}
		//Remove function type from templated kernel
		if(k.find(' ') != std::string::npos) {
			k.replace(0, 1 + k.find_last_of(' '), "");
		}
	}

	std::ofstream out(k + ".DivSpill.csv");

	if(!out.is_open()) return;

	out << "div;cst;divLD;CstLD;divST;ConstST" << std::endl
			<< divSpills << ';' << cstSpills << ';' << divLoads << ';'
			<< cstLoads << ';' << divStores << ';' << cstStores;
	
	out.close();
}

}

#endif

namespace transforms
{

bool DivergenceRegister::Constant  = false;
bool DivergenceRegister::Divergent = true;

analysis::DataflowGraph::RegisterId DivergenceRegister::warpPosition = 0;

bool DivergenceRegister::divergent() const
{
	return _state;
}

DivergenceRegister::DivergenceRegister(RegisterId ireg, Type itype,
	MemoryArray* local, MemoryArray* shared, const bool state) :
	CoalescedRegister(ireg, itype, local), _shared(shared)
{

}

void DivergenceRegister::combineState(const bool state)
{
	_state |= state;
}

void DivergenceRegister::load(DataflowGraph & dfg,
	InstructionList &il, RegisterId &dreg)
{
	report("Load coalesced variable " << _reg << ", SSA variable " << dreg);
	if(_state)
	{
#if DIVERGENCE_REGISTER_PROFILE_H_
		divergenceProfiler::divLoads++;
#endif
		report("\tGoes into local memory");
		CoalescedRegister::load(dfg, il, dreg);
		return;
	}

#if DIVERGENCE_REGISTER_PROFILE_H_
	divergenceProfiler::cstLoads++;
#endif

	if(predicate())
	{
		report("\tIs a constant predicate");
		ir::PTXInstruction ld( ir::PTXInstruction::Opcode::Ld );
		ld.type = predType;
		ld.addressSpace = _shared->addressSpace();
		ld.a = ir::PTXOperand( ir::PTXOperand::AddressMode::Address,
			predType, _shared->name(), _shared->getVarOffset(_reg) );
		ld.d = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
			predType, dfg.newRegister());
		il.push_back(ld);
		report("ld:		" << ld.toString() );

		ir::PTXInstruction setp( ir::PTXInstruction::Opcode::SetP );
		setp.type = predType;
		setp.a = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
			predType, ld.d.reg );
		setp.d = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
			_type, dreg );
		setp.b = ir::PTXOperand( 1, predType);
		setp.comparisonOperator = ir::PTXInstruction::CmpOp::Eq;
		report("setp:		" << setp.toString() );
		il.push_back(setp);
		return;
	}

	report("\tIs a constant variable");
	ir::PTXInstruction ld( ir::PTXInstruction::Opcode::Ld );
	ld.type = _type;
	ld.addressSpace = _shared->addressSpace();
	ld.a = ir::PTXOperand( ir::PTXOperand::AddressMode::Address,
		_type, _shared->name(), _shared->getVarOffset(_reg) );
	ld.d = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
		_type, dreg );
	il.push_back(ld);
	report("ld:		" << ld.toString() );
}



void DivergenceRegister::store(DataflowGraph &dfg,
	InstructionList &il, RegisterId &sreg) {
	
	report("Affine store for variable " << sreg
		<< ", coalesced to " << _reg << ", state: " << _state);
		
	if(_state)
	{
#if DIVERGENCE_REGISTER_PROFILE_H_
		divergenceProfiler::divStores++;
#endif
		CoalescedRegister::store(dfg, il, sreg);
		return;
	}
	
#if DIVERGENCE_REGISTER_PROFILE_H_
	divergenceProfiler::cstStores++;
#endif

	if(predicate())
	{
		report("\tConstant predicate");
		ir::PTXInstruction selp( ir::PTXInstruction::Opcode::SelP );
		selp.type = predType;
		selp.a = ir::PTXOperand( 1, predType);
		selp.b = ir::PTXOperand( 0, predType);
		selp.c = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
			_type, sreg );
		selp.d = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
			predType, dfg.newRegister() );
		report("selp:		" << selp.toString() );
		il.push_back(selp);

		ir::PTXInstruction st( ir::PTXInstruction::Opcode::St );
		st.type = predType;
		st.addressSpace = _shared->addressSpace();
		st.d = ir::PTXOperand( ir::PTXOperand::AddressMode::Address,
			predType, _shared->name(), _shared->getVarOffset(_reg) );
		st.a = ir::PTXOperand( ir::PTXOperand::AddressMode::Register,
			predType, selp.d.reg );
		report("st:		" << st.toString() );
		il.push_back(st);
		return;
	}
	
	report("\tConstant variable");
	ir::PTXInstruction st( ir::PTXInstruction::Opcode::St );
	st.type = _type;
	st.addressSpace = _shared->addressSpace();
	st.d = ir::PTXOperand( ir::PTXOperand::AddressMode::Address, _type,
		_shared->name(), _shared->getVarOffset(_reg) );
	st.a = ir::PTXOperand( ir::PTXOperand::AddressMode::Register, _type, sreg );
	report("st:		" << st.toString() );
	il.push_back(st);
}

void DivergenceRegister::spill()
{
	if(_spilled)
		return;
	if(_state)
	{
#if DIVERGENCE_REGISTER_PROFILE_H_
		divergenceProfiler::divSpills++;
#endif
		CoalescedRegister::spill();
		return;
	}
	
#if DIVERGENCE_REGISTER_PROFILE_H_
	divergenceProfiler::cstSpills++;
#endif

	_spilled = true;
	
	if(_type == ir::PTXOperand::DataType::pred)
	{
		_shared->insertVar(_reg, predType);
	}
	else
	{
		_shared->insertVar(_reg, _type);
	}
}

}

