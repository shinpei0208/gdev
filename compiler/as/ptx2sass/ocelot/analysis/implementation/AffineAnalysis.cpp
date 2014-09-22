/*! \file AffineAnalysis.h
 \date May 21, 2011
 \author Diogo Sampaio <dnsampaio@gmail.com>
 \author Fernando Pereira <fpereira@dcc.ufmg.br>
 \author Sylvain Collange <sylvain.collange@gmail.com>
 \brief The source file for the AffineAnalysis class
 */

#include <ocelot/analysis/interface/AffineAnalysis.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis 
{

using ir::Instruction;
using ir::PTXInstruction;
using ir::PTXOperand;
using ir::ControlFlowGraph;

bool AffineTransferFunction::_isFunct = false;
// ConstantAbstractState ---------------------------

ConstantAbstractState ConstantAbstractState::Meet(
	ConstantAbstractState const & other) const {
	ConstantAbstractState dest(State(std::min(type, other.type)), 0);
	switch(type){
		case Bottom:
		break;
		case Top:
			dest.value = other.value;
		break;
		case Constant:
			if(other.type == Top ||
				(other.type == Constant && value == other.value)){
				dest.value = value;
			}else{
				dest.type = Bottom;
			}
		break;
		default:
			assert(false);
		break;
	}
	return dest;
}

ConstantAbstractState const & ConstantAbstractState::operator+=(ConstantAbstractState const & other) {
	return *this = ConstantAbstractState(State(std::min(type, other.type)), value + other.value);
}

ConstantAbstractState const & ConstantAbstractState::operator-=(ConstantAbstractState const & other) {
	return *this = ConstantAbstractState(State(std::min(type, other.type)), value - other.value);
}
ConstantAbstractState const & ConstantAbstractState::operator*=(ConstantAbstractState const & other) {
	ConstantAbstractState r(State(std::min(type, other.type)), value * other.value);
	if((type == Constant && value == 0) || (other.type == Constant && other.value == 0)){
		// Zero times anything is zero
		r.type = Constant;
		r.value = 0;
	}
	return *this = r;
}

ConstantAbstractState const ConstantAbstractState::top = ConstantAbstractState(ConstantAbstractState::Top, 0);
ConstantAbstractState const ConstantAbstractState::bottom = ConstantAbstractState(ConstantAbstractState::Bottom, 0);
ConstantAbstractState const ConstantAbstractState::zero = ConstantAbstractState(ConstantAbstractState::Constant, 0);
ConstantAbstractState const ConstantAbstractState::one = ConstantAbstractState(ConstantAbstractState::Constant, 1);

std::ostream & operator<<(std::ostream & s, ConstantAbstractState const & as) {
	switch(as.type){
		case ConstantAbstractState::Bottom:
			s << "B";
		break;
		case ConstantAbstractState::Top:
			s << "T";
		break;
		case ConstantAbstractState::Constant:
			s << as.value;
		break;
		default:
			assert(false);
		break;
	}
	return s;
}

// AffineAbstractState --------------------------

AffineAbstractState AffineAbstractState::Meet(AffineAbstractState const & other) const {
	return AffineAbstractState(base.Meet(other.base), stride[0].Meet(other.stride[0]), stride[1].Meet(other.stride[1]));
}

AffineAbstractState const AffineAbstractState::top = AffineAbstractState(ConstantAbstractState::top,
		ConstantAbstractState::top, ConstantAbstractState::top);
AffineAbstractState const AffineAbstractState::bottom = AffineAbstractState(ConstantAbstractState::bottom,
		ConstantAbstractState::bottom, ConstantAbstractState::bottom);
AffineAbstractState const AffineAbstractState::uniform = AffineAbstractState(ConstantAbstractState::bottom,
		ConstantAbstractState::zero, ConstantAbstractState::zero);

AffineAbstractState const & AffineAbstractState::operator+=(AffineAbstractState const & other) {
	base += other.base;
	stride[0] += other.stride[0];
	stride[1] += other.stride[1];
	return *this;
}

AffineAbstractState const & AffineAbstractState::operator-=(AffineAbstractState const & other) {
	base -= other.base;
	stride[0] -= other.stride[0];
	stride[1] -= other.stride[1];
	return *this;
}

AffineAbstractState const & AffineAbstractState::operator*=(AffineAbstractState const & other) {
	// (aX+b)*(cX+d) = acX^2 + (bc+ad)X + bd
	ConstantAbstractState x2 = (stride[0] * other.stride[0]) + (stride[1] * other.base) + (other.stride[1] * base);
	ConstantAbstractState x = stride[0] * other.base + base * other.stride[0];
	ConstantAbstractState c = base * other.base;
	if(x2 != ConstantAbstractState::zero){
		// We only support affine relations, not higher-order
		*this = bottom;
	}else{
		base = c;
		stride[0] = x;
		stride[1] = x2;
	}
	return *this;
}

// Non-linear operator
// If stride of both inputs is zero, stride of output is zero
AffineAbstractState const & AffineAbstractState::operator^=(AffineAbstractState const & other) {
	AffineAbstractState r = Meet(other);
	return *this = r.NonLinear();
}
AffineAbstractState AffineAbstractState::NonLinear() const {
	AffineAbstractState r = *this;
	r.base = ConstantAbstractState::bottom;
	if((r.stride[0] != ConstantAbstractState::zero && r.stride[0] != ConstantAbstractState::top)
			|| (r.stride[1] != ConstantAbstractState::zero && r.stride[1] != ConstantAbstractState::top)){
		r.stride[0] = ConstantAbstractState::bottom;
		r.stride[1] = ConstantAbstractState::bottom;
	}
	return r;
}

AffineAbstractState const & AffineAbstractState::operator&=(AffineAbstractState const & other) {
	return *this = Meet(other);
}

AffineAbstractState Meet(AffineAbstractState const & a, AffineAbstractState const & b) {
	return a.Meet(b);
}

std::ostream & operator<<(std::ostream & s, AffineAbstractState const & as) {
	s << "(" << as.stride[1] << "+" << as.stride[0] << "+" << as.base << ")";
	return s;
}

// AffineTransferFunction ------------------------------

AffineTransferFunction::AffineTransferFunction(StateMap & m, const DataflowGraph* dfg) :
		states(m), _dfg(dfg) {
}

AffineAbstractState AffineTransferFunction::operandState(ir::PTXOperand const & op) {
	//TODO: This is a simplistic treatment of vector types, should treat each vector variable independently
	if(!op.array.empty()){
		AffineAbstractState state = AffineAbstractState::top;
		for(ir::PTXOperand::Array::const_iterator reg = op.array.begin(); reg != op.array.end(); reg++){
			state.Meet(states[reg->reg]);
		}
		switch(op.type){
			case PTXOperand::DataType::f16:
			case PTXOperand::DataType::f32:
			case PTXOperand::DataType::f64:
				if(state.affine())
					return AffineAbstractState::bottom;
				if(state.isUniform())
					return AffineAbstractState::uniform;
			break;
			default:
				if(op.imm_int == const_t(op.imm_int)){
					return AffineAbstractState(const_t(op.imm_int));
				}
			break;
		}
		return state;
	}

	switch(op.addressMode){
		case PTXOperand::AddressMode::Register:
			return registerState(op);
		case PTXOperand::AddressMode::Indirect:
			return states[op.reg] + AffineAbstractState(op.offset);
		case PTXOperand::AddressMode::Immediate:
			switch(op.type){
				case PTXOperand::DataType::f16:
				case PTXOperand::DataType::f32:
				case PTXOperand::DataType::f64:
					return AffineAbstractState::uniform;
				default:
					if(op.imm_int == const_t(op.imm_int))
						return AffineAbstractState(const_t(op.imm_int));
				break;
			}
			return AffineAbstractState::uniform;
		case PTXOperand::AddressMode::Invalid:
			return AffineAbstractState::top;
		case PTXOperand::AddressMode::Special:
			return specialState(op);
		case PTXOperand::AddressMode::Address:
			return AffineAbstractState::uniform;
		case PTXOperand::AddressMode::Label:
			return AffineAbstractState::top; // Just for convenience and get proper graph colors
		default:
			cerr << "Affine analysis: add addressMode " << PTXOperand::toString(op.addressMode) << endl;
		break;
	}
	return AffineAbstractState::bottom;
}

AffineAbstractState AffineTransferFunction::registerState(ir::PTXOperand const & op) {
	switch(op.type){
		case PTXOperand::DataType::f16:
		case PTXOperand::DataType::f32:
		case PTXOperand::DataType::f64:
			// Hack: the way PTX poorly differentiate between int and FP instructions
			// makes it easier to detect FP at operand level than at instruction level.
			// BUG: this causes wrong colors in graph output (type of mov dest?)...
			return states[op.reg].NonLinear();
		case PTXOperand::DataType::pred:
			if(op.condition == PTXOperand::PredicateCondition::PT || op.condition == ir::PTXOperand::PredicateCondition::nPT)
				return AffineAbstractState::top;
			else
				return states[op.reg].NonLinear();
		default:
			return states[op.reg];
	}
}

AffineAbstractState AffineTransferFunction::specialState(ir::PTXOperand const & op) {
	//TODO: Propagate laneId and tid.[yz] information
	if(op.special == PTXOperand::SpecialRegister::laneId){
		return AffineAbstractState::bottom;
	}
	if(op.special == PTXOperand::SpecialRegister::tid){
		if(op.vIndex == PTXOperand::VectorIndex::ix){
			return AffineAbstractState(ConstantAbstractState::zero, ConstantAbstractState::one, ConstantAbstractState::zero);
		}
	}
	return AffineAbstractState::uniform;
}

void AffineTransferFunction::readInputStates(ir::PTXInstruction const & ptxi, AffineAbstractState types[]) {
	// 0: pg
	// 1: a
	// 2: b
	// 3: c
	// 4: d
	for(int i = 0; i != 5; ++i){
		types[i] = AffineAbstractState::top;
	}

	PTXOperand PTXInstruction::* sources[5] = { &PTXInstruction::pg, &PTXInstruction::a, &PTXInstruction::b,
			&PTXInstruction::c, &PTXInstruction::d };

	unsigned int limit = 4;

	switch(ptxi.opcode){
		case ir::PTXInstruction::Opcode::St:
			limit = 5;
		break;
		default:
		break;
	}
	for(unsigned int j = 0; j < limit; ++j){
		ir::PTXOperand const & op = ptxi.*sources[j];
		types[j] = operandState(op);
	}
}

bool AffineTransferFunction::operator()(DataflowGraph::Instruction const & insn) {
	if(insn.d.empty()){
		return false; // Nothing to do, nothing changes
	}

	bool stillMoving = false;

	assert(insn.i->ISA == Instruction::Architecture::PTX);
	// We only support PTX, for now
	PTXInstruction const & ptxi = static_cast<PTXInstruction const &>(*insn.i);

	AffineAbstractState outputState;
	AffineAbstractState inputStates[5];

	if(!(ptxi.a.array.empty() && ptxi.b.array.empty() && ptxi.c.array.empty())){
		DataflowGraph::RegisterPointerVector::const_iterator a = insn.s.begin();
		DataflowGraph::RegisterPointerVector::const_iterator b = insn.s.end();
		for(; a != b; a++){
			inputStates[1] ^= states[*a->pointer];
		}
	}else{
		readInputStates(ptxi, inputStates);
	}

	switch(ptxi.opcode){
		// Non-linear (breaks affine property), but keeps uniform property
		case PTXInstruction::Opcode::And:
		case PTXInstruction::Opcode::CNot:
		case PTXInstruction::Opcode::Cos:
		case PTXInstruction::Opcode::Div:
		case PTXInstruction::Opcode::Ex2:
		case PTXInstruction::Opcode::Lg2:
		case PTXInstruction::Opcode::Max:
		case PTXInstruction::Opcode::Min:
		case PTXInstruction::Opcode::Not:
		case PTXInstruction::Opcode::Or:
		case PTXInstruction::Opcode::Rcp:
		case PTXInstruction::Opcode::Rem:
		case PTXInstruction::Opcode::Rsqrt:
		case PTXInstruction::Opcode::Sad:
		case PTXInstruction::Opcode::Sin:
		case PTXInstruction::Opcode::Sqrt:
		case PTXInstruction::Opcode::Xor:
		case PTXInstruction::Opcode::Clz:
		case PTXInstruction::Opcode::Ldu:
		case PTXInstruction::Opcode::Tex:
			outputState = inputStates[1] ^ inputStates[2] ^ inputStates[3];
		break;

			// Linear, keeps affine, uniform property (except FP)
		case PTXInstruction::Opcode::Cvt:
		case PTXInstruction::Opcode::Mov:
			outputState = inputStates[1];
		break;
		case PTXInstruction::Opcode::Add:
		case PTXInstruction::Opcode::AddC:
			// Rule SUM
			outputState = inputStates[1] + inputStates[2];
		break;
		case PTXInstruction::Opcode::Sub:
		case PTXInstruction::Opcode::SubC:
			outputState = inputStates[1] - inputStates[2];
		break;
		case PTXInstruction::Opcode::Set: // hack...
		case PTXInstruction::Opcode::SetP:
			outputState = inputStates[1] - inputStates[2] - inputStates[3];
		break;
		case PTXInstruction::Opcode::Abs:
			outputState = inputStates[1];
			outputState.base.value = std::abs(outputState.base.value);
			outputState.stride[0].value = std::abs(outputState.stride[0].value);
			outputState.stride[1].value = std::abs(outputState.stride[1].value);
		break;
		case PTXInstruction::Opcode::Neg:
			outputState = inputStates[1];
			outputState.base.value = -outputState.base.value;
			outputState.stride[0].value = -outputState.stride[0].value;
			outputState.stride[1].value = -outputState.stride[1].value;
		break;
		case PTXInstruction::Opcode::Ld:
			// LD.LOCAL has thread-local address space
			if((ptxi.addressSpace == PTXInstruction::AddressSpace::Local)
					|| ((ptxi.addressSpace == PTXInstruction::AddressSpace::Param) && (_isFunct))){
				outputState = AffineAbstractState::bottom;
			}else{
				outputState = inputStates[1] ^ inputStates[2] ^ inputStates[3];
			}
		break;
		case PTXInstruction::Opcode::Mul:
		case PTXInstruction::Opcode::Mul24:
			outputState = inputStates[1] * inputStates[2];
		break;

			// Could improve accuracy on the following instructions
			// with additional rules
		case PTXInstruction::Opcode::Mad24:
		case PTXInstruction::Opcode::Mad:
			outputState = inputStates[1] * inputStates[2] + inputStates[3];
		break;
		case PTXInstruction::Opcode::Shl:
			outputState = inputStates[1] << inputStates[2];
		break;
		case PTXInstruction::Opcode::SelP:
		case PTXInstruction::Opcode::Shr:
		case PTXInstruction::Opcode::SlCt:
		case PTXInstruction::Opcode::Vote:
		case PTXInstruction::Opcode::Fma:
			outputState = inputStates[1] ^ inputStates[2] ^ inputStates[3];
			if(!outputState.isUniform())
				outputState = AffineAbstractState::bottom;
		break;

		case PTXInstruction::Opcode::Cvta:
			if(ptxi.addressSpace == PTXInstruction::AddressSpace::Local)
				outputState = AffineAbstractState::bottom;
			else
				outputState = inputStates[1] ^ inputStates[2] ^ inputStates[3];
		break;

			// TODO: Add newer PTX instructions
		case PTXInstruction::Opcode::Atom:
			outputState = AffineAbstractState::bottom;
		break;
		default:
			cerr << "Affine analysis: add instruction " << PTXInstruction::toString(ptxi.opcode) << endl;
		break;
	}

	// Guard predicate
	outputState = Meet(outputState, inputStates[0]);

	setStateDests(insn.d, outputState, stillMoving);
	return stillMoving;
}

bool AffineTransferFunction::operator()(DataflowGraph::PhiInstruction const & phi) {
	/*TODO: Check this
	 * If all sources of a phi instruction have the exact same
	 * known value, they are not divergent */
	bool canDiverge = false;
	report(""<< phi << " +++++");
	if(_dfg->hasPhi(&phi)){
		typedef set<DataflowGraph::RegisterId> IDSet;
		IDSet const & preds = _dfg->getPhiPredicates(&phi);

		for(IDSet::const_iterator it = preds.begin(); !(canDiverge) && (it != preds.end()); ++it){
			report("	 depends on predicate " << *it << "	==	" << states[*it]);
			if((states[*it].stride[0] == ConstantAbstractState::bottom)
					|| ((states[*it].stride[1] != ConstantAbstractState::top)
							&& (states[*it].stride[1] != ConstantAbstractState::zero))){
				canDiverge = true;
			}
		}
	}
#if REPORT_BASE
	if (canDiverge)
	report("	 and can diverge");
#endif
	if(canDiverge && (phi.s.size() == 1)){
		report("		and is life split, setting " << phi.d.id << " as divergent");
		return setState(phi.d.id, AffineAbstractState::bottom);
	}

	AffineAbstractState t = AffineAbstractState::top;

	for(DataflowGraph::RegisterVector::const_iterator it = phi.s.begin(); it != phi.s.end(); ++it){
		t = t.Meet(states[it->id]);
	}
	report("		has joined state " << t);
	if(canDiverge){
		if((t.base == ConstantAbstractState::bottom) || (t.stride[0] == ConstantAbstractState::bottom)){
			report("		that has bottom base or stride, so it's divergent");
			return setState(phi.d.id, AffineAbstractState::bottom);
		}
	}
	report("		that is kept to " << phi.d.id);
	return setState(phi.d.id, t);
}

bool AffineTransferFunction::setState(DataflowGraph::RegisterId id, AffineAbstractState state) {
	if(states.find(id) == states.end() || (AffineAbstractState(states[id]) != state)){
		states[id] = state;
		return true;
	}else{
		return false;
	}
}

void AffineTransferFunction::setStateDests(DataflowGraph::RegisterPointerVector const & d, AffineAbstractState state,
	bool & stillMoving) {
	for(DataflowGraph::RegisterPointerVector::const_iterator it = d.begin(); it != d.end(); ++it){
		// predicates cannot be Affine
		if(it->type == PTXOperand::DataType::pred)
			stillMoving |= setState(*it->pointer, state.NonLinear());
		else
			stillMoving |= setState(*it->pointer, state);
	}
}

void AffineTransferFunction::dumpMap(std::ostream & os) const {
	for(StateMap::const_iterator it = states.begin(); it != states.end(); ++it){
		os << it->first << " -> " << it->second << std::endl;
	}
}

// AffineAnalysis ------------------------

/*! \brief Constructor. Initalized the fields of the object.
 */
AffineAnalysis::AffineAnalysis() :
	KernelAnalysis("DataflowGraphAnalysis",
	{"DominatorTreeAnalysis", "PostDominatorTreeAnalysis"})

{
	_sa = NULL;
	_aft = NULL;
}

/*! \brief Analyze the control and data flow graphs searching for divergent
 *		variables and blocks
 */
void AffineAnalysis::analyze(ir::IRKernel &k) {
	_kernel = &k;
	AffineTransferFunction::_isFunct = _kernel->function();
	auto dfg = static_cast<DataflowGraph*>(getAnalysis("DataflowGraphAnalysis"));

	dfg->convertToSSAType(DataflowGraph::Gated);

	_sm = new AffineTransferFunction::StateMap;
	_aft = new AffineTransferFunction(*_sm, dfg);
	
	_sa = new SparseAnalysis(dfg, _aft);
	_sa->iterateToFixPoint();
}

}

