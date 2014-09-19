/*!
	\file SharedPtrAttribute.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Sept 1, 2012
	\brief updates kernels by adding the shared memory base ptr
*/

// C++ includes
#include <set>

// Ocelot includes
#include <ocelot/ir/interface/Module.h>
#include <ocelot/transforms/interface/SharedPtrAttribute.h>

// Hydrazine Includes
#include <hydrazine/interface/Version.h>
#include <hydrazine/interface/debug.h>

////////////////////////////////////////////////////////////////////////////////

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////////////////////////

transforms::SharedPtrAttribute::SharedPtrAttribute(const std::string &n) {

}

transforms::SharedPtrAttribute::~SharedPtrAttribute() {

}

void transforms::SharedPtrAttribute::runOnModule(ir::Module& m) {
	std::string externName = _getOrInsertExternShared(m);
	assert(externName != "");
	
	for (ir::Module::KernelMap::const_iterator k_it = m.kernels().begin(); 
		k_it != m.kernels().end(); ++k_it ) {
		_updateSharedPtrUses(*k_it->second, externName);
	}
}

std::string transforms::SharedPtrAttribute::_getOrInsertExternShared(ir::Module &m) {
	typedef ir::Module::GlobalMap GlobalMap;
	
	for (GlobalMap::const_iterator glb_it = m.globals().begin(); glb_it != m.globals().end(); ++glb_it) {
		if (glb_it->second.space() == ir::PTXInstruction::Shared && 
			glb_it->second.statement.attribute == ir::PTXStatement::Extern) {
			return glb_it->second.name();
		}
	}
	
	ir::PTXStatement statement(ir::PTXStatement::Shared);
	statement.type = ir::PTXOperand::u8;
	statement.name = "externSharedBasePtr";
	statement.attribute = ir::PTXStatement::Extern;

	ir::Global global(statement);
	m.insertGlobal(global);
	return statement.name;
}

static ir::PTXOperand _newRegister(ir::PTXKernel &k, ir::PTXOperand::DataType addrType, ir::PTXOperand::RegisterType &reg) {
	return ir::PTXOperand(ir::PTXOperand::Register, addrType, reg++);
}

void transforms::SharedPtrAttribute::_updateSharedPtrUses(ir::PTXKernel &k, std::string symbol) {

	std::map<std::string, ir::PTXOperand::RegisterType > mappedRegister;
	for (ir::Kernel::ParameterVector::iterator it = k.arguments.begin(); 
		it != k.arguments.end(); ++it) {
		
		if (it->isPtrDeclaration() && it->ptrAddressSpace == ir::PTXInstruction::Shared) {
			mappedRegister[it->name] = 0;
		}
	}
	
	ir::PTXOperand::DataType addressType = ir::PTXOperand::u64;
	ir::PTXOperand::RegisterType newRegister = k.getUnusedRegister();
	
	ir::PTXInstruction movSymbol(ir::PTXInstruction::Mov);
	
	movSymbol.d = _newRegister(k, addressType, newRegister);
	movSymbol.a = ir::PTXOperand(ir::PTXOperand::Address, addressType, symbol);
	movSymbol.type = addressType;
	ir::BasicBlock::InstructionList instructions;
	instructions.push_back(new ir::PTXInstruction(movSymbol));
	
	std::string entryBlockLabel = (*k.cfg()->begin()).label();
	
	for (auto it = mappedRegister.begin(); it != mappedRegister.end(); ++it) {
		ir::PTXInstruction ld(ir::PTXInstruction::Ld);
		ld.d = _newRegister(k, addressType, newRegister);
		ld.a = ir::PTXOperand(ir::PTXOperand::Indirect, it->first); // .ptr .shared parameter
		ld.type = addressType;
		ld.addressSpace = ir::PTXInstruction::Param;
		
		// insert it
		instructions.push_back(new ir::PTXInstruction(ld));
		
		// add the value to the base
		ir::PTXInstruction add(ir::PTXInstruction::Add);
		add.type = addressType;
		add.d = _newRegister(k, addressType, newRegister);
		add.a = ld.d;
		add.b = movSymbol.d;
		
		// insert it
		instructions.push_back(new ir::PTXInstruction(add));
		
		mappedRegister[it->first] = add.d.reg;
	}
	
	for (auto it = instructions.begin(); it != instructions.end(); ++it) {
		report("inserting instruction: " << (*it)->toString());
	}
	k.cfg()->begin()->instructions.insert(k.cfg()->begin()->instructions.begin(), 
		instructions.begin(), instructions.end());
	
	// replace uses of the parameter with a move
	for (ir::ControlFlowGraph::iterator bb_it = k.cfg()->begin(); bb_it != k.cfg()->end(); ++bb_it) {
		for (ir::BasicBlock::instruction_iterator inst_it = bb_it->instructions.begin(); 
			inst_it != bb_it->instructions.end(); ++inst_it) {
			ir::PTXInstruction *inst = static_cast<ir::PTXInstruction *>(*inst_it);
			
			if (bb_it->label() == entryBlockLabel) {
				bool skip = false;
				for (auto s_it = instructions.begin(); s_it != instructions.end(); ++s_it) {
					if (*s_it == inst) {
						skip = true;
						break;
					}
				}
				if (skip) {
					continue;
				}
			}
			
			if (inst->opcode == ir::PTXInstruction::Ld && 
				inst->addressSpace == ir::PTXInstruction::Param && 
				mappedRegister.find(inst->a.identifier) != mappedRegister.end()) {
				inst->opcode = ir::PTXInstruction::Mov;
				inst->a = ir::PTXOperand(ir::PTXOperand::Register, addressType, 
					mappedRegister[inst->a.identifier]);
			}
		}
	}
}

bool transforms::SharedPtrAttribute::testModule(const ir::Module &m) {
	for (ir::Module::KernelMap::const_iterator kernel = m.kernels().begin(); 
		kernel != m.kernels().end(); ++kernel) {
	
		for (ir::Kernel::ParameterVector::const_iterator it = kernel->second->arguments.begin(); 
			it != kernel->second->arguments.end(); ++it) {
		
			if (it->isPtrDeclaration() && it->ptrAddressSpace == ir::PTXInstruction::Shared) {
				return true;
			}
		}
	}
	return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

