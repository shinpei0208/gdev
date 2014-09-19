/*! \file EmulatedKernel.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 19, 2009
	\brief implements the Kernel base class
*/

// Ocelot includes
#include <ocelot/api/interface/OcelotConfiguration.h>

#include <ocelot/executive/interface/EmulatedKernel.h>
#include <ocelot/executive/interface/EmulatorDevice.h>
#include <ocelot/executive/interface/RuntimeException.h>
#include <ocelot/executive/interface/CooperativeThreadArray.h>
#include <ocelot/executive/interface/EmulatedKernelScheduler.h>

#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/Parameter.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

#include <ocelot/trace/interface/TraceGenerator.h>

#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/transforms/interface/IPDOMReconvergencePass.h>
#include <ocelot/transforms/interface/ThreadFrontierReconvergencePass.h>
#include <ocelot/transforms/interface/DefaultLayoutPass.h>
#include <ocelot/transforms/interface/EnforceLockStepExecutionPass.h>
#include <ocelot/transforms/interface/PriorityLayoutPass.h>


// C++ includes
#include <cassert>
#include <cmath>
#include <vector>
#include <map>
#include <unordered_set>
#include <cstring>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

////////////////////////////////////////////////////////////////////////////////

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define REPORT_KERNEL_INSTRUCTIONS  0
#define REPORT_LAUNCH_CONFIGURATION 1

////////////////////////////////////////////////////////////////////////////////


static unsigned int align(unsigned int offset, unsigned int _size) {
	unsigned int size = _size == 0 ? 1 : _size;
	unsigned int difference = offset % size;
	unsigned int alignedOffset = difference == 0 
		? offset : offset + size - difference;
	return alignedOffset;
}

executive::EmulatedKernel::EmulatedKernel(
	ir::IRKernel* kernel, 
	Device* d, 
	bool _initialize) 
: 
	ExecutableKernel(*kernel, d),
	CTA(0),
	_initialized(false)
{
	report("Created emulated kernel " << name);
	assertM(kernel->ISA == ir::Instruction::PTX, 
		"Can only build an emulated kernel from a PTXKernel.");
	
	ISA = ir::Instruction::Emulated;
	ConstMemory = ArgumentMemory = 0;
	if (_initialize) {
		initialize();
	}
}

executive::EmulatedKernel::EmulatedKernel(
	Device* d): ExecutableKernel(d), CTA(0), _initialized(false) {
	ISA = ir::Instruction::Emulated;
}

executive::EmulatedKernel::EmulatedKernel(): CTA(0), _initialized(false) {
	ISA = ir::Instruction::Emulated;
}

executive::EmulatedKernel::~EmulatedKernel() {
	freeAll();
}

bool executive::EmulatedKernel::executable() const {
	report("EmulatedKernel::executable() returns true");
	return true;
}

void executive::EmulatedKernel::launchGrid(int width, int height, int depth) {
	report("EmulatedKernel::launchGrid called for " << name);
	report("  " << _registerCount << " registers");

	initializeSymbolReferences();

	_gridDim = ir::Dim3(width, height, depth);	
	
	// notify trace generator(s)
	initializeTraceGenerators();

#if REPORT_LAUNCH_CONFIGURATION == 1
	report("EmulatedKernel::launchGrid(" << width << ", " << height << ")");
	report("  kernel: " << name);
	report("  const:  " << constMemorySize() << " bytes");
	report("  local:  " << localMemorySize() << " bytes");
	report("  static shared: " << sharedMemorySize() << " bytes");
	report("  extern shared: " << externSharedMemorySize() << " bytes");
	report("  total shared:  " << totalSharedMemorySize() << " bytes");
	report("  argument: " << argumentMemorySize() << " bytes");
	report("  param: " << parameterMemorySize() << " bytes");
	report("  max threads: " << maxThreadsPerBlock() << " threads per block");
	report("  registers: " << registerCount() << " registers");
	report("  grid: " << gridDim().x << ", " << gridDim().y
		<< ", " << gridDim().z);
	report("  block: " << blockDim().x << ", " << blockDim().y
		<< ", " << blockDim().z);
#endif

	EmulatedKernelScheduler kernelScheduler(
		static_cast<EmulatorDevice*>(device));
	
	kernelScheduler.launch(this, gridDim(), &_generators);
	
	finalizeTraceGenerators();
}

void executive::EmulatedKernel::setKernelShape(int x, int y, int z) {
	_blockDim.x = x;
	_blockDim.y = y;
	_blockDim.z = z;
}

ir::Dim3 executive::EmulatedKernel::getKernelShape() const {
	return _blockDim;
}

void executive::EmulatedKernel::setExternSharedMemorySize(unsigned int bytes) {
	report("Setting external shared memory size to " << bytes);
	_externSharedMemorySize = bytes;
}

void executive::EmulatedKernel::setWorkerThreads(unsigned int limit) {
}


void executive::EmulatedKernel::setExternalFunctionSet(
	const ir::ExternalFunctionSet& s) {
	_externals = &s;
}

void executive::EmulatedKernel::clearExternalFunctionSet() {
	_externals = 0;
}


void executive::EmulatedKernel::freeAll() {
	delete [] ConstMemory;
	delete [] ArgumentMemory;
	ArgumentMemory = ConstMemory = 0;
}

void executive::EmulatedKernel::initialize() {
	if (!_initialized) {
		_initialized = true;
		registerAllocation();
		constructInstructionSequence();
		initializeTextureMemory();
		initializeSharedMemory();
		initializeArgumentMemory();
		initializeStackMemory();
		updateParamReferences();
		initializeLocalMemory();
		initializeGlobalLocalMemory();
		invalidateCallTargets();
	}
}
		
void executive::EmulatedKernel::constructInstructionSequence() {

	report("Constructing emulated instruction sequence.");

	// This kernel/function begins at the first instruction
	functionEntryPoints.insert(std::make_pair(name, 0));

	transforms::PassManager manager(const_cast<ir::Module*>(module));
	
	typedef api::OcelotConfiguration config;

	if (config::get().executive.reconvergenceMechanism
		== ReconvergenceMechanism::Reconverge_IPDOM) {
		
		transforms::IPDOMReconvergencePass* pass
			= new transforms::IPDOMReconvergencePass;

		manager.addPass(pass);
		manager.runOnKernel(*this);

		instructions = std::move(pass->instructions);
	}
	else if (config::get().executive.reconvergenceMechanism
		== ReconvergenceMechanism::Reconverge_Barrier) {
		// just pack the instructions into a vector
		transforms::DefaultLayoutPass* pass
			= new transforms::DefaultLayoutPass;

		manager.addPass(pass);
		manager.runOnKernel(*this);

		instructions = std::move(pass->instructions);
	}
	else if (config::get().executive.reconvergenceMechanism
		== ReconvergenceMechanism::Reconverge_TFSortedStack) {

		transforms::ThreadFrontierReconvergencePass* pass
			= new transforms::ThreadFrontierReconvergencePass(false);

		manager.addPass(pass);
		manager.runOnKernel(*this);

		instructions = std::move(pass->instructions);
	}
	else if (config::get().executive.reconvergenceMechanism
		== ReconvergenceMechanism::Reconverge_TFGen6) {

		transforms::ThreadFrontierReconvergencePass* pass
			= new transforms::ThreadFrontierReconvergencePass(true);

		manager.addPass(pass);
		manager.runOnKernel(*this);

		instructions = std::move(pass->instructions);
	}
	else if (config::get().executive.reconvergenceMechanism
		== ReconvergenceMechanism::Reconverge_TFSoftware)
	{
		transforms::PriorityLayoutPass* layout
			= new transforms::PriorityLayoutPass();

		manager.addPass(layout);
		manager.runOnKernel(*this);
		
		instructions = std::move(layout->instructions);
	}
	else {
		assertM(false, "unknown thread reconvergence mechanism - "
			<< ReconvergenceMechanism::toString(
				(ReconvergenceMechanism::Type)
				config::get().executive.reconvergenceMechanism));
	}
}

/*!
	After emitting the instruction sequence, visit each memory move operation 
	and replace references to parameters with offsets into parameter memory.

	Data movement instructions: ld, st
*/
void executive::EmulatedKernel::updateParamReferences() {
	using namespace std;
	for (PTXInstructionVector::iterator 
		i_it = instructions.begin();
		i_it != instructions.end(); ++i_it) {
		ir::PTXInstruction& instr = *i_it;
		if (instr.addressSpace == ir::PTXInstruction::Param) {
			if (instr.opcode == ir::PTXInstruction::Ld 
				&& instr.a.addressMode == ir::PTXOperand::Address) {
				
				ir::Parameter *pParam = getParameter(instr.a.identifier);
				if (pParam) {
					ir::Parameter &param = *pParam;
					instr.a.isArgument = param.isArgument() && !function();
					report("For instruction '" << instr.toString() 
							<< "' setting source parameter '"
							<< instr.a.toString() 
							<< "' offset to "
							<< (param.offset + instr.a.offset) << " " 
							<< ( instr.a.isArgument ? "(argument)" : "" ) );
					instr.a.offset += param.offset;
					instr.a.imm_uint = 0;
				}
			}
			else if (instr.opcode == ir::PTXInstruction::St
				&& instr.d.addressMode == ir::PTXOperand::Address) {
				ir::Parameter *pParam = getParameter(instr.d.identifier);
				if (pParam) {
					ir::Parameter &param = *pParam;
					instr.d.isArgument = param.isArgument() && !function();
					report("For instruction '" << instr.toString() 
							<< "' setting destination parameter '"
							<< instr.d.toString() 
							<< "' offset to "
							<< (instr.d.offset + param.offset) << " " 
							<< ( instr.d.isArgument ? "(argument)" : "" ) );

					instr.d.offset += param.offset;
					instr.d.imm_uint = 0;
				}
			}
		}
	}
}

void executive::EmulatedKernel::initializeArgumentMemory() {
	report( "Initializing argument memory for kernel " << name );
	delete[] ArgumentMemory;
	ArgumentMemory = 0;
	_argumentMemorySize = 0;

	if(!function()) {
		for(ParameterVector::iterator i_it = arguments.begin();
			i_it != arguments.end(); ++i_it) {
			ir::Parameter& argument = *i_it;
			// align parameter memory
			unsigned int padding = argument.getAlignment() 
				- ( _argumentMemorySize % argument.getAlignment() );
			padding = (argument.getAlignment() == padding) ? 0 : padding;
		
			report("  offset: " << _argumentMemorySize << ", alignment: "
				<< argument.getAlignment() << ", padding: " << padding);
			_argumentMemorySize += padding;
			argument.offset = _argumentMemorySize;
		
			report( " Initializing memory for argument " << argument.name 
				<< " of size " << argument.getSize() << " at offset "
				<< argument.offset << " with " << padding
				<< " bytes padding from previous element" );
			
			_argumentMemorySize += argument.getSize();
		}
	
		ArgumentMemory = new char[_argumentMemorySize];
	}
	
	report(" Total argument size is " << argumentMemorySize());
}

bool executive::EmulatedKernel::checkMemoryAccess(const void* base, 
	size_t size) const {
	if(device == 0) return false;
	return device->checkMemoryAccess(base, size);
}

void executive::EmulatedKernel::updateArgumentMemory() {
	using namespace std;

	if(!function()) {
		unsigned int size = 0;
		for(ParameterVector::iterator i_it = arguments.begin();
			i_it != arguments.end(); ++i_it) {
			ir::Parameter& argument = *i_it;
			unsigned int padding = argument.getAlignment()
				- (size % argument.getAlignment());
			padding = (argument.getAlignment() == padding) ? 0 : padding;
			
			report("  offset: " << size << ", alignment: "
				<< argument.getAlignment() << ", padding: " << padding);
			
			size += padding;
			
			for(ir::Parameter::ValueVector::iterator
				v_it = argument.arrayValues.begin();
				v_it != argument.arrayValues.end(); ++v_it) {

				report( " updating memory for argument " << argument.name 
					<< " of size " << argument.getSize() << " at offset "
					<< argument.offset );
			
				assert(size < _argumentMemorySize);
				memcpy(ArgumentMemory + size, &v_it->val_b16,
					argument.getElementSize());
				size += argument.getElementSize();
			}
		}
	}
	
	// skip parameters because they cannot have initial values
}

void executive::EmulatedKernel::updateMemory() {
	updateGlobals();
}

executive::ExecutableKernel::TextureVector 
	executive::EmulatedKernel::textureReferences() const {
	return textures;
}


void executive::EmulatedKernel::registerAllocation() {
	using namespace std;
	report("Allocating registers");
	registerMap = ir::PTXKernel::assignRegisters( *cfg() );
	_registerCount = registerMap.size();
	report(" Allocated " << _registerCount << " registers");
}

/*!
	Allocates arrays in shared memory and maps identifiers to allocations.
*/
void executive::EmulatedKernel::initializeSharedMemory() {
	using namespace std;
	typedef std::unordered_map<string, unsigned int> Map;
	typedef std::unordered_map<std::string, 
		ir::Module::GlobalMap::const_iterator> GlobalMap;
	typedef std::
	unordered_set<std::string> StringSet;
	typedef std::deque<ir::PTXOperand*> OperandVector;
	unsigned int sharedOffset = 0;
	unsigned int externalAlignment = 1;

	report( "Initializing shared memory for kernel " << name );
	Map label_map;
	GlobalMap sharedGlobals;
	StringSet external;
	OperandVector externalOperands;
	
	if(module != 0) {
		for(ir::Module::GlobalMap::const_iterator 
			it = module->globals().begin(); 
			it != module->globals().end(); ++it) {
			if (it->second.statement.directive == ir::PTXStatement::Shared) {
				if(it->second.statement.attribute == ir::PTXStatement::Extern) {
					report("Found global external shared variable " 
						<< it->second.statement.name);
					assert(external.count(it->second.statement.name) == 0);
					external.insert(it->second.statement.name);
					externalAlignment = std::max( externalAlignment, 
						(unsigned int) it->second.statement.accessAlignment() );
					externalAlignment = std::max( externalAlignment, 
						ir::PTXOperand::bytes( it->second.statement.type ) );
				} 
				else {
					report("Found global shared variable " 
						<< it->second.statement.name);
					sharedGlobals.insert( std::make_pair( 
						it->second.statement.name, it ) );
				}
			}
		}
	}
	
	LocalMap::const_iterator it = locals.begin();
	for (; it != locals.end(); ++it) {
		if (it->second.space == ir::PTXInstruction::Shared) {
			if(it->second.attribute == ir::PTXStatement::Extern) {
				report("Found local external shared variable " 
					<< it->second.name);
				assert(external.count(it->second.name) == 0);
					external.insert(it->second.name);
				externalAlignment = std::max( externalAlignment, 
					(unsigned int) it->second.getAlignment() );
				externalAlignment = std::max( externalAlignment, 
					ir::PTXOperand::bytes( it->second.type ) );
			}
			else {
				unsigned int offset;

				ir::PTXKernel::computeOffset(it->second.statement(), offset, sharedOffset);
				label_map[it->second.name] = offset;
				report("Found local shared variable " << it->second.name 
					<< " at offset " << offset << " with alignment " 
					<< it->second.getAlignment() << " of size " 
					<< (sharedOffset - offset ));
			}
		}
	}

	// now visit every instruction and change the address mode from 
	// label to immediate, and assign the offset as an immediate value 
	ir::PTXOperand ir::PTXInstruction:: *operands[] = { &ir::PTXInstruction::d,
		&ir::PTXInstruction::a, &ir::PTXInstruction::b, &ir::PTXInstruction::c
	};

	bool hasCalls = false;

	PTXInstructionVector::iterator i_it = instructions.begin();
	for (; i_it != instructions.end(); ++i_it) {
		ir::PTXInstruction &instr = *i_it;

		hasCalls |= instr.isCall();

		// look for mov and ld/st instructions
		if (instr.mayHaveAddressableOperand()) {
			for (int n = 0; n < 4; n++) {
				if ((instr.*operands[n]).addressMode 
					== ir::PTXOperand::Address) {
					StringSet::iterator si = external.find(
						(instr.*operands[n]).identifier);
					if (si != external.end()) {
						externalOperands.push_back(&(instr.*operands[n]));
						continue;
					}
					
					GlobalMap::iterator gi = sharedGlobals.find(
							(instr.*operands[n]).identifier);
					if (gi != sharedGlobals.end()) {
						ir::Module::GlobalMap::const_iterator 
							it = gi->second;
						sharedGlobals.erase(gi);
						unsigned int offset;

						report("Found global shared variable " 
							<< it->second.statement.name);
						ir::PTXKernel::computeOffset(it->second.statement, 
							offset, sharedOffset);						
						label_map[it->second.statement.name] = offset;
					}
					
					Map::iterator l_it 
						= label_map.find((instr.*operands[n]).identifier);
					if (label_map.end() != l_it) {
						(instr.*operands[n]).type = ir::PTXOperand::u64;
						(instr.*operands[n]).imm_uint = l_it->second;
						report("For instruction " << instr.toString() 
							<< ", mapping shared label " << l_it->first 
							<< " to " << l_it->second);
					}
				}
			}
		}
	}
	
	// compute necessary padding for alignment of external shared memory
	unsigned int padding = externalAlignment 
		- (sharedOffset % externalAlignment);
	padding = (padding == externalAlignment) ? 0 : padding;
	sharedOffset += padding;

	report("Padding shared memory by " << padding << " bytes to handle " 
		<< externalAlignment << " byte alignment requirement.");
		
	for (OperandVector::iterator operand = externalOperands.begin(); 
		operand != externalOperands.end(); ++operand) {
		report( "Mapping external shared label " << (*operand)->identifier 
			<< " to " << sharedOffset );
		(*operand)->type = ir::PTXOperand::u64;
		(*operand)->imm_uint = sharedOffset;
	}

	// allocate shared memory object
	_sharedMemorySize = sharedOffset;
	
	if (hasCalls) {
		_sharedMemorySize = std::max(_sharedMemorySize,
			_getSharedMemorySizeOfReachableKernels());
	}

	report("Total shared memory size is " << _sharedMemorySize);
}

/*!
	Allocates arrays in local memory and maps identifiers to allocations.
*/
void executive::EmulatedKernel::initializeLocalMemory() {
	using namespace std;

	unsigned int localOffset = 0;

	map<string, unsigned int> label_map;
	
	report("Initialize local memory");
		
	LocalMap::const_iterator it = locals.begin();
	for (; it != locals.end(); ++it) {
		if (it->second.space == ir::PTXInstruction::Local) {
			unsigned int offset;

			report("  found local local variable " 
				<< it->second.name);
			ir::PTXKernel::computeOffset(it->second.statement(), offset, localOffset);						
			label_map[it->second.name] = offset;
		}
	}

	ir::PTXOperand ir::PTXInstruction:: *operands[] = {&ir::PTXInstruction::d,
		&ir::PTXInstruction::a, &ir::PTXInstruction::b, &ir::PTXInstruction::c};
	PTXInstructionVector::iterator 
		i_it = instructions.begin();
	for (; i_it != instructions.end(); ++i_it) {
		ir::PTXInstruction &instr = *i_it;

		// look for instructions that can reference addresses
		if (instr.mayHaveAddressableOperand()) {
			for (int n = 0; n < 4; n++) {
				if ((instr.*operands[n]).addressMode 
					== ir::PTXOperand::Address) {
					map<string, unsigned int>::iterator 
						l_it = label_map.find((instr.*operands[n]).identifier);
					if (label_map.end() != l_it) {
						(instr.*operands[n]).isGlobalLocal = false;
						(instr.*operands[n]).type = ir::PTXOperand::u64;
						(instr.*operands[n]).imm_uint = l_it->second;
						report("  for instruction " << instr.toString() 
							<< ", mapping local label " << l_it->first 
							<< " to " << l_it->second);
					}
				}
			}
		}
	}

	// allocate local memory object
	_localMemorySize = localOffset;
	report(" Total local memory size " << _localMemorySize);
}

void executive::EmulatedKernel::initializeGlobalLocalMemory() {
	unsigned int localOffset = 0;

	std::map<std::string, unsigned int> label_map;
	
	report("Initialize global local memory");
	
	if(module != 0) {
		for(ir::Module::GlobalMap::const_iterator 
			it = module->globals().begin(); 
			it != module->globals().end(); ++it) {
			if (it->second.statement.directive == ir::PTXStatement::Local) {
				unsigned int offset;

				report(" Found globally scoped local variable " 
					<< it->second.statement.name);
				ir::PTXKernel::computeOffset(it->second.statement, 
					offset, localOffset);						
				label_map[it->second.statement.name] = offset;
			}
		}
	}

	ir::PTXOperand ir::PTXInstruction:: *operands[] = {&ir::PTXInstruction::d,
		&ir::PTXInstruction::a, &ir::PTXInstruction::b, &ir::PTXInstruction::c};
	PTXInstructionVector::iterator 
		i_it = instructions.begin();
	for (; i_it != instructions.end(); ++i_it) {
		ir::PTXInstruction &instr = *i_it;

		// look for instructions that can reference addresses
		if (instr.mayHaveAddressableOperand()) {
			for (int n = 0; n < 4; n++) {
				if ((instr.*operands[n]).addressMode 
					== ir::PTXOperand::Address) {
					std::map<std::string, unsigned int>::iterator 
						l_it = label_map.find((instr.*operands[n]).identifier);
					if (label_map.end() != l_it) {
						(instr.*operands[n]).isGlobalLocal = true;
						(instr.*operands[n]).type = ir::PTXOperand::u64;
						(instr.*operands[n]).imm_uint = l_it->second;
						report(" For instruction " << instr.toString() 
							<< ", mapping globally scoped local " << l_it->first 
							<< " to " << l_it->second);
					}
				}
			}
		}
	}

	report(" Total globally scoped local memory size is " << localOffset);
	// get size for global local memory object
	_globalLocalMemorySize = localOffset;
}


/*! Maps identifiers to const memory allocations. */
void executive::EmulatedKernel::initializeConstMemory() {
	using namespace std;
	assert(module != 0);

	report("Initializing constant variables for kernel " << name);

	unsigned int constantOffset = 0;

	typedef map<string, unsigned int> ConstantOffsetMap;

	ConstantOffsetMap constant;
	ir::Module::GlobalMap::const_iterator it = module->globals().begin();
	for (; it != module->globals().end(); ++it) {
		if (it->second.statement.directive == ir::PTXStatement::Const) {
			unsigned int offset;

			report("  Found global const variable " 
				<< it->second.statement.name);
			ir::PTXKernel::computeOffset(it->second.statement, 
				offset, constantOffset);						
			constant[it->second.statement.name] = offset;
		}
	}
	
	report( "Total constant memory size is " << constantOffset );

	ir::PTXOperand ir::PTXInstruction:: *operands[] = {
		&ir::PTXInstruction::d, &ir::PTXInstruction::a, &ir::PTXInstruction::b, 
		&ir::PTXInstruction::c
	};
	PTXInstructionVector::iterator i_it = instructions.begin();
	for (; i_it != instructions.end(); ++i_it) {
		ir::PTXInstruction &instr = *i_it;

		// look for mov instructions or ld/st instruction
		if (instr.mayHaveAddressableOperand()) {
			for (int n = 0; n < 4; n++) {
				if ((instr.*operands[n]).addressMode 
					== ir::PTXOperand::Address) {
					ConstantOffsetMap::iterator	l_it 
						= constant.find((instr.*operands[n]).identifier);
					if (constant.end() != l_it) {
						report("For instruction " << instr.toString() 
							<< ", mapping constant label " << l_it->first 
							<< " to " << l_it->second );
						(instr.*operands[n]).type = ir::PTXOperand::u64;
						(instr.*operands[n]).imm_uint = l_it->second;
					}
				}
			}
		}
	}

	// allocate constant memory object
	delete[] ConstMemory;
	
	_constMemorySize = constantOffset;
	if (_constMemorySize > 0) {
		ConstMemory = new char[_constMemorySize];
	}
	else {
		ConstMemory = 0;
	}
	
	// copy globals into constant memory
	for (ConstantOffsetMap::iterator l_it = constant.begin(); 
		l_it != constant.end(); ++l_it) {

		assert(device != 0);
		Device::MemoryAllocation* global = device->getGlobalAllocation(
			module->path(), l_it->first);

		assert(global != 0);
		assert(global->size() + l_it->second <= _constMemorySize);

		memcpy(ConstMemory + l_it->second, global->pointer(), global->size());
	}

}

/*!
	Maps identifiers to global memory allocations.
*/
void executive::EmulatedKernel::initializeGlobalMemory() {
	using namespace std;
	if(module == 0) return;
	
	report("Initializing global variables for kernel " << name);

	unordered_set<string> global;
	
	ir::Module::GlobalMap::const_iterator it = module->globals().begin();
	for (; it != module->globals().end(); ++it) {
		if (it->second.statement.directive == ir::PTXStatement::Global) {
			report(" Found global variable " << it->second.statement.name);
			global.insert(it->second.statement.name);
		}
	}

	// now visit every instruction and change the address mode from label 
	// to immediate, and assign the offset as an immediate value 
	ir::PTXOperand ir::PTXInstruction:: *operands[] = {
		&ir::PTXInstruction::d, &ir::PTXInstruction::a, &ir::PTXInstruction::b, 
		&ir::PTXInstruction::c };
	PTXInstructionVector::iterator 
		i_it = instructions.begin();
	for (; i_it != instructions.end(); ++i_it) {
		ir::PTXInstruction &instr = *i_it;

		// look for mov instructions or ld/st/atom instruction
		if (instr.mayHaveAddressableOperand()) {
			for (int n = 0; n < 4; n++) {
				if ((instr.*operands[n]).addressMode 
					== ir::PTXOperand::Address) {
					unordered_set<string>::iterator l_it = 
						global.find((instr.*operands[n]).identifier);
					if (global.end() != l_it) {
						(instr.*operands[n]).type = ir::PTXOperand::u64;
						assert( device != 0);
						Device::MemoryAllocation* allocation = 
							device->getGlobalAllocation(
							module->path(), *l_it);
						assert(allocation != 0);
						(instr.*operands[n]).imm_uint = 
							(ir::PTXU64)allocation->pointer();
						report("Mapping global label " 
							<< (instr.*operands[n]).identifier << " to " 
							<< (void *)(instr.*operands[n]).imm_uint 
							<< " for instruction " << instr.toString() );
					}
				}
			}
		}
	}
}

void executive::EmulatedKernel::setCTA(CooperativeThreadArray* cta) {
	CTA = cta;
}

void executive::EmulatedKernel::fixBranchTargets(size_t basePC) {
	for (size_t pc = basePC; pc != instructions.size(); ++pc) {
		ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
			instructions[pc]);
		
		ptx.pc = pc;
		
		if(ptx.opcode == ir::PTXInstruction::Bra
			|| ptx.opcode == ir::PTXInstruction::Call
			|| (ptx.opcode == ir::PTXInstruction::Mov
				&& ptx.a.addressMode == ir::PTXOperand::FunctionName)) {
			if(ptx.branchTargetInstruction != -1) {
				ptx.branchTargetInstruction += basePC;
			}
		}
		
		if(ptx.opcode == ir::PTXInstruction::Bra) {
			if(!ptx.uni) {
				ptx.reconvergeInstruction += basePC;
			}
		}
		
	}
}

size_t executive::EmulatedKernel::link(const std::string& functionName) {
	report("Getting PC for kernel '" << functionName << "'");
	EmulatedKernel* kernel = static_cast<EmulatedKernel*>(
		device->getKernel(module->path(), functionName));
	assertM(kernel != 0, "Kernel function '" << functionName 
		<< "' not found in module '" << module->path() << "'");
	FunctionNameMap::iterator entryPoint =
		functionEntryPoints.find(functionName);

	if (entryPoint == functionEntryPoints.end()) {
		int newPC = instructions.size();
		report(" linking kernel '" << functionName << "' at pc " << newPC);
		kernelEntryPoints.insert(std::make_pair(newPC, kernel));
		
		kernel->updateGlobals();
		
		std::string name = functionName;
		
		instructions.insert(instructions.end(), kernel->instructions.begin(), 
			kernel->instructions.end());
		
		fixBranchTargets(newPC);
		
		entryPoint = functionEntryPoints.insert(
			std::make_pair(name, newPC)).first;
	}
	
	return entryPoint->second;
}

void executive::EmulatedKernel::lazyLink(int callPC, 
	const std::string& functionName) {
	std::string name = functionName;
	
	int pc = link(name);
	
	instructions[callPC].branchTargetInstruction = pc;
	
	if (instructions[callPC].opcode == ir::PTXInstruction::Call) {
		EmulatedKernel* kernel = static_cast<EmulatedKernel*>(
				device->getKernel(module->path(), name));
		assertM(kernel != 0, "Kernel function '" << name 
			<< "' not found in module '" << module->path() << "'");
	
		instructions[callPC].a.stackMemorySize = 
			kernel->parameterMemorySize();
		instructions[callPC].a.localMemorySize = kernel->localMemorySize();
		instructions[callPC].a.sharedMemorySize = kernel->sharedMemorySize();
		instructions[callPC].a.registerCount = kernel->registerCount();
	}
}

const executive::EmulatedKernel* 
	executive::EmulatedKernel::getKernel(int PC) const {
	report("Getting kernel at pc " << PC);
	PCToKernelMap::const_iterator kernel = kernelEntryPoints.find(PC);
	if (kernel == kernelEntryPoints.end()) return 0;
	return kernel->second;
}

const executive::EmulatedKernel* 
	executive::EmulatedKernel::getKernelContainingThisPC(int PC) const {
	report("Getting kernel containing pc " << PC);
	
	for(auto kernel = kernelEntryPoints.begin();
		kernel != kernelEntryPoints.end(); ++kernel) {
		if(PC < kernel->first) continue;
		if((unsigned int)PC >=
			kernel->first + kernel->second->instructions.size()) {
			continue;
		}
		
		return kernel->second;
	}
	
	if((unsigned int)PC < instructions.size()) return this;
	
	return 0;
}

void executive::EmulatedKernel::jumpToPC(int PC) {
	assert(CTA != 0);
	
	CTA->jumpToPC(PC);
}

executive::EmulatedKernel::RegisterFile 
	executive::EmulatedKernel::getCurrentRegisterFile() const {
	assert(CTA != 0);
	return CTA->getCurrentRegisterFile();
}

const char* executive::EmulatedKernel::getSharedMemory() const {
	assert(CTA != 0);
	return (char*) CTA->functionCallStack.sharedMemoryPointer();
}

const char* executive::EmulatedKernel::getLocalMemory(unsigned int tid) const {
	assert(CTA != 0);
	return (char*) CTA->functionCallStack.localMemoryPointer(tid);
}

const char* executive::EmulatedKernel::getGlobalLocalMemory(
	unsigned int tid) const {
	assert(CTA != 0);
	return (char*) CTA->functionCallStack.globalLocalMemoryPointer(tid);
}

unsigned int
	executive::EmulatedKernel::getCurrentFrameArgumentMemorySize() const {
	assert(CTA != 0);
	return CTA->functionCallStack.previousFrameSize();
}

unsigned int
	executive::EmulatedKernel::getCurrentFrameLocalMemorySize() const {
	assert(CTA != 0);
	return CTA->functionCallStack.localMemorySize();
}

unsigned int
	executive::EmulatedKernel::getCurrentFrameParameterMemorySize() const {
	assert(CTA != 0);
	return CTA->functionCallStack.stackFrameSize();
}

const char* executive::EmulatedKernel::getStackBase(
	unsigned int threadId) const {
	assert(CTA != 0);
	return (const char*)CTA->functionCallStack.stackBase();
}

unsigned int executive::EmulatedKernel::getTotalStackSize(
	unsigned int threadId) const {
	assert(CTA != 0);
	return CTA->functionCallStack.totalStackSize();
}

unsigned int executive::EmulatedKernel::getStackFrameCount() const {
	assert(CTA != 0);
	return CTA->functionCallStack.getFrameCount();
}

executive::FrameInfo executive::EmulatedKernel::getStackFrameInfo(
	unsigned int frame) const {
	assert(CTA != 0);
	return CTA->functionCallStack.getFrameInfo(frame);
}

void executive::EmulatedKernel::initializeStackMemory() {
	_parameterMemorySize = 0;
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;

	report("Initializing stack memory for kernel " << name);
	
	OffsetMap offsets;

	if(function()) {
		unsigned int offset = 0;
		for(ParameterVector::iterator i_it = arguments.begin();
			i_it != arguments.end(); ++i_it) {
			ir::Parameter& parameter = *i_it;
			// align parameter memory
			offset = align(offset, parameter.getAlignment());
			parameter.offset = offset;
			report( " Initializing memory for stack parameter " 
				<< parameter.name 
				<< " of size " << parameter.getSize() << " at offset " 
				<< offset );
			offset += parameter.getSize();
		}
		
		_parameterMemorySize = std::max(_parameterMemorySize, offset);
	}
	
	unsigned int callParameterStackBase = _parameterMemorySize;
	
	report(" Setting offsets of operands to call instructions.");
	for (PTXInstructionVector::iterator fi = instructions.begin(); 
		fi != instructions.end(); ++fi) {
		
		if (fi->opcode == ir::PTXInstruction::Call) {
			report( "  For '" << fi->toString() << "'" );
			unsigned int offset = 0;

			fi->b.offset = callParameterStackBase;
		
			for (ir::PTXOperand::Array::iterator argument = fi->d.array.begin(); 
				argument != fi->d.array.end(); ++argument) {

				if(argument->isRegister())
				{
					unsigned int size = ir::PTXOperand::bytes(argument->type);
					
					offset = align(offset, size);
					argument->offset = offset;
					report("   For return argument '" << argument->toString() 
						<< "' stack offset " << offset);

					offsets.insert(std::make_pair(argument->identifier, offset));

					offset += size;
				}
				else
				{
					auto parameter = parameters.find(argument->identifier);
					assert(parameter != parameters.end());

					offset = align(offset, parameter->second.getSize());
					argument->offset = offset;
					report("   For return argument '" << argument->identifier 
						<< "' stack offset " << offset << " -> argument offset " 
						<< argument->offset);

					offsets.insert(std::make_pair(argument->identifier, offset));

					offset += parameter->second.getSize();
				}
			}
			
			for (ir::PTXOperand::Array::iterator argument = fi->b.array.begin(); 
				argument != fi->b.array.end(); ++argument) {
				if(argument->isRegister())
				{
					unsigned int size = ir::PTXOperand::bytes(argument->type);
					
					offset = align(offset, size);
					argument->offset = offset;
					report("   For call argument '" << argument->toString() 
						<< "' stack offset " << offset);

					offsets.insert(std::make_pair(argument->identifier, offset));

					offset += size;
				}
				else
				{
					auto parameter = parameters.find(argument->identifier);
					assert(parameter != parameters.end());

					offset = align(offset, parameter->second.getSize());
					argument->offset = offset;
					report("   For call argument '" << argument->identifier 
						<< "' stack offset " << offset << " -> argument offset " 
						<< argument->offset);

					offsets.insert(std::make_pair(argument->identifier, offset));

					offset += parameter->second.getSize();
				}
			}
			
			_parameterMemorySize = std::max(_parameterMemorySize,
				callParameterStackBase + offset);
		}
	}
	
	for (ParameterMap::iterator i_it = parameters.begin();
		i_it != parameters.end(); ++i_it) {
		ir::Parameter& parameter = i_it->second;
		
		OffsetMap::iterator offset = offsets.find(parameter.name);
		if(offset != offsets.end())
		{
			parameter.offset = offset->second;
		
			report( " Setting offset of stack parameter " << parameter.name 
				<< " of size " << parameter.getSize() << " to " 
				<< offset->second );
		}
	}
	
	report(" Parameter stack memory requirement is " << _parameterMemorySize);
}

void executive::EmulatedKernel::initializeTextureMemory() {
	typedef std::unordered_map<std::string, unsigned int> IndexMap;
	if(module == 0) return;

	textures.clear();
	IndexMap indices;

	unsigned int next = 0;

	for (PTXInstructionVector::iterator fi = instructions.begin(); 
		fi != instructions.end(); ++fi) {
		if (fi->opcode == ir::PTXInstruction::Tex) {
			assert(device != 0);
			ir::Texture* texture = (ir::Texture*)device->getTextureReference(
				module->path(), fi->a.identifier);
			assert(texture != 0);
			
			IndexMap::iterator index = indices.find(fi->a.identifier);

			if (index == indices.end()) {
				index = indices.insert(std::make_pair(fi->a.identifier,
					next++)).first;
				textures.push_back(texture);
			}

			fi->a.reg = index->second;
			report("updated fi->a.reg = " << fi->a.reg);
		}
	}

	report("Registered indices:");
	#if(REPORT_BASE > 0)
	for (IndexMap::const_iterator ind_it = indices.begin(); 
		ind_it != indices.end(); ++ind_it) {
		report("  " << ind_it->first << ": " << ind_it->second 
			<< " - type: " << textures[ind_it->second]->type 
			<< " - data: " << textures[ind_it->second]->data);
	}
	#endif
}

void executive::EmulatedKernel::initializeSymbolReferences() {
	report("Initializing symbol references stored in global variables.");
	
	if (module != 0) {
		for (ir::Module::GlobalMap::const_iterator 
			it = module->globals().begin(); 
			it != module->globals().end(); ++it) {
			for (ir::PTXStatement::SymbolVector::const_iterator symbol =
				it->second.statement.array.symbols.begin(); symbol !=
				it->second.statement.array.symbols.end(); ++symbol) {
				Device::MemoryAllocation* allocation = 
					device->getGlobalAllocation(module->path(),
					it->second.name());
				assert(allocation != 0);
				
				size_t pc = link(symbol->name);
				
				size_t size = ir::PTXOperand::bytes(it->second.statement.type);
				size_t offset = symbol->offset * size;
				
				report(" Setting symbol '" << symbol->name << "' to " << pc
					<< " in global '" << it->second.name() << "' at offset "
					<< offset);
				std::memcpy((char*)allocation->pointer() + offset, &pc, size);
			}
		}
	}
	
}

void executive::EmulatedKernel::invalidateCallTargets() {
	report( "Invalidating call instruction targets." );
	for (PTXInstructionVector::iterator fi = instructions.begin(); 
		fi != instructions.end(); ++fi) {
		if (fi->opcode == ir::PTXInstruction::Mov) {
			if (fi->a.addressMode == ir::PTXOperand::FunctionName) {
				report( " For '" << fi->toString() << "'" );
				fi->branchTargetInstruction = -1;
			}
		}
		else if (fi->opcode == ir::PTXInstruction::Call) {
			report( " For '" << fi->toString() << "'" );
			fi->branchTargetInstruction = -1;
		}
	}
}

void executive::EmulatedKernel::updateGlobals() {
	initializeConstMemory();
	initializeGlobalMemory();
}

std::string executive::EmulatedKernel::toString() const {
	std::stringstream stream;
	stream << "Kernel " << name << "\n";
	for( PTXInstructionVector::const_iterator 
		fi = instructions.begin(); 
		fi != instructions.end(); ++fi ) {
		const ir::PTXInstruction &instr = *fi;
		stream << "[PC " << fi - instructions.begin() << "] " << 
			fi->toString();
		if (instr.opcode == ir::PTXInstruction::Bra) {
			stream << " [target: " << instr.branchTargetInstruction 
				<< ", reconverge: " << instr.reconvergeInstruction << "]";
		}
		stream << "\n";
	}
	return stream.str();
}

std::string executive::EmulatedKernel::fileName() const {
	assert(module != 0);
	return module->path();
}

std::string executive::EmulatedKernel::location( unsigned int PC ) const {
	assert(module != 0 );
	assert(PC < instructions.size());
	unsigned int statement = instructions[PC].statementIndex;
	
	while (statement == (unsigned int) -1) {
		if (PC >= instructions.size()) {
			statement = 0;
			break;
		}
		
		statement = instructions[++PC].statementIndex;
	}
	
	auto s_it = module->statements().begin();
	std::advance(s_it, statement);
	auto s_rit = ir::Module::StatementVector::const_reverse_iterator(s_it);
	unsigned int program = 0;
	unsigned int line = 0;
	unsigned int col = 0;
	for ( ; s_rit != module->statements().rend(); ++s_rit) {
		if (s_rit->directive == ir::PTXStatement::Loc) {
			line = s_rit->sourceLine;
			col = s_rit->sourceColumn;
			program = s_rit->sourceFile;
			break;
		}
	}
	
	std::string fileName;
	for ( s_it = module->statements().begin(); 
		s_it != module->statements().end(); ++s_it ) {
		if (s_it->directive == ir::PTXStatement::File) {
			if (s_it->sourceFile == program) {
				fileName = s_it->name;
				break;
			}
		}
	}
	
	std::stringstream stream;
	stream << fileName << ":" << line << ":" << col;
	return stream.str();
}

std::string executive::EmulatedKernel::getInstructionBlock(int PC) const {
	
	auto bt_it = basicBlockMap.lower_bound(PC);
	
	if (bt_it != basicBlockMap.end()) {
		return bt_it->second;
	}
	
	return "";
}

/*! \brief accessor for obtaining PCs of first and last instructions in a block */
std::pair<int,int> executive::EmulatedKernel::getBlockRange(
	const std::string &label) const { 
	return blockPCRange.at(label); 
}

unsigned int executive::EmulatedKernel::_getSharedMemorySizeOfReachableKernels() const {
	unsigned int size = 0;
	
	auto kernels = static_cast<EmulatorDevice*>(device)->getAllKernels();

	for(auto kernel = kernels.begin(); kernel != kernels.end(); ++kernel) {
		size = std::max(size, (*kernel)->sharedMemorySize());
	}

	return size;
}

