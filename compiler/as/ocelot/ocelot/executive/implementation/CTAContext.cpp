/*! \file CTAContext.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements a CTAContext
*/

#include <ocelot/executive/interface/CTAContext.h>
#include <ocelot/executive/interface/EmulatedKernel.h>
#include <ocelot/executive/interface/CooperativeThreadArray.h>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////

executive::CTAContext::CTAContext(
	const executive::EmulatedKernel *k, 
	executive::CooperativeThreadArray *c)
		: PC(0), executionState(Running), kernel(k), cta(c)
{
	
	ir::Dim3 blockDim = cta->blockDim;
	active = boost::dynamic_bitset<>(blockDim.x * blockDim.y * blockDim.z, 1);

	for (int i = 0; i < blockDim.x*blockDim.y*blockDim.z; i++) {
		active[i] = 1;
	}

	report("CTAContext(0x" << std::hex << (unsigned long)k << ", 0x" 
		<< (unsigned long)c << ")" << std::dec);
}

executive::CTAContext::~CTAContext() {

}

////////////////////////////////////////////////////////////////////////////////
		
bool executive::CTAContext::predicated(int threadID,
	const ir::PTXInstruction &instr) {
	using namespace ir;
	
	bool on = false;
	if (active[threadID]) {
		ir::PTXOperand::PredicateCondition condition = instr.pg.condition;
		switch (condition) {
		case PTXOperand::PT:
			on = true;
			break;
		case PTXOperand::nPT:
			on = false;
			break;
		default:
			{
				bool pred = cta->getRegAsPredicate(threadID, instr.pg.reg);
				on = ((pred && condition == PTXOperand::Pred) 
					|| (!pred && condition == PTXOperand::InvPred));
			}
			break;
		}
	}
	return on;
}

boost::dynamic_bitset<> executive::CTAContext::predicateMask(
	const ir::PTXInstruction &instr) {
	boost::dynamic_bitset<> result(active.size(), false);
	
	int threads = result.size();
	for (int i = 0; i < threads; ++i) {
		result[i] = predicated(i, instr);
	}
	
	return result;
}

bool executive::CTAContext::running() const {
	return executionState == Running;
}

////////////////////////////////////////////////////////////////////////////////

