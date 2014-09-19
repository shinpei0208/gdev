/*! \file ATIExecutableKernel.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 10, 2010
 *  \brief The source file for the ATI Executable Kernel class.
 */

// Ocelot includes
#include <ocelot/executive/interface/ATIExecutableKernel.h>
#include <ocelot/translator/interface/PTXToILTranslator.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Exception.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define Throw(x) {std::stringstream s; s << x; \
	throw hydrazine::Exception(s.str()); }

namespace executive
{
	ATIExecutableKernel::ATIExecutableKernel(ir::IRKernel &k,
			CALcontext *context,
			CALevent *event, CALresource *uav0, CALresource *cb0, 
			CALresource *cb1, Device* d)
		: 
			ExecutableKernel(k, d), 
			_voteMemorySize(0),
			_context(context),
			_event(event),
			_info(),
			_module(0), 
			_object(0), 
			_image(0), 
			_uav0Resource(uav0),
			_uav0Mem(0),
			_uav0Name(0),
			_uav8Name(0),
			_cb0Resource(cb0), 
			_cb0Mem(0),
			_cb0Name(0),
			_cb1Resource(cb1),
			_cb1Mem(0),
			_cb1Name(0)
	{
		registerAllocation();
		initializeSharedMemory();
	}

	unsigned int ATIExecutableKernel::_pad(size_t& size, unsigned int alignment)
	{
		unsigned int padding = alignment - (size % alignment);
		padding = (alignment == padding) ? 0 : padding;
		size += padding;
		return padding;
	}

	void ATIExecutableKernel::registerAllocation() {
		//using namespace std;
		report("Allocating registers");
		registerMap = ir::PTXKernel::assignRegisters( *cfg() );

        // HACK: we need to hard-code the number of hw registers since
        // registerMap only refers to virtual ptx registers
		//_registerCount = registerMap.size();
        _registerCount = 10;
		report(" Allocated " << _registerCount << " registers");
	}

	void ATIExecutableKernel::initializeSharedMemory()
	{
		report("Allocating shared memory");

		typedef std::unordered_map<std::string, size_t> AllocationMap;
		typedef std::unordered_set<std::string> StringSet;
		typedef std::deque<ir::PTXOperand*> OperandVector;
		typedef std::unordered_map<std::string,
				ir::Module::GlobalMap::const_iterator> GlobalMap;

		AllocationMap map;
		GlobalMap sharedGlobals;
		StringSet external;
		OperandVector externalOperands;

		unsigned int externalAlignment = 1;
		size_t sharedSize = 0;

		assert(module != 0);

		// global shared variables
		ir::Module::GlobalMap globals = module->globals();
		ir::Module::GlobalMap::const_iterator global;
		for (global = globals.begin() ; global != globals.end() ; global++)
		{
			ir::PTXStatement statement = global->second.statement;
			if (statement.directive == ir::PTXStatement::Shared)
			{
				if (statement.attribute == ir::PTXStatement::Extern)
				{
					report("Found global external shared variable \""
							<< statement.name << "\"");

					assertM(external.count(statement.name) == 0,
							"External global \"" << statement.name
							<< "\" declared more than once.");

					external.insert(statement.name);
					externalAlignment = std::max(externalAlignment,
							(unsigned int)statement.alignment);
					externalAlignment = std::max(externalAlignment,
							ir::PTXOperand::bytes(statement.type));
				} else {
					report("Found global shared variable \"" 
							<< statement.name << "\"");
					sharedGlobals.insert(
							std::make_pair(statement.name, global));
				}
			}
		}

		// local shared variables	
		LocalMap::const_iterator local;
		for (local = locals.begin() ; local != locals.end() ; local++)
		{
			if (local->second.space == ir::PTXInstruction::Shared)
			{
				if (local->second.attribute == ir::PTXStatement::Extern)
				{
					report("Found local external shared variable \"" 
							<< local->second.name << "\"");

					assertM(external.count(local->second.name) == 0,
							"External local \"" << local->second.name
							<< "\" declared more than once.");

					external.insert(local->second.name);
					externalAlignment = std::max(externalAlignment,
							(unsigned int)local->second.alignment);
					externalAlignment = std::max(externalAlignment,
							ir::PTXOperand::bytes(local->second.type));
				} else
				{
					report("Allocating local shared variable \""
							<< local->second.name << "\" of size "
							<< local->second.getSize());

					_pad(sharedSize, local->second.alignment);
					map.insert(std::make_pair(local->second.name, sharedSize));
					sharedSize += local->second.getSize();
				}
			}
		}

		ir::ControlFlowGraph::iterator block;
		for (block = cfg()->begin() ; block != cfg()->end() ; block++)
		{
			ir::ControlFlowGraph::InstructionList insts = block->instructions;
			ir::ControlFlowGraph::InstructionList::iterator inst;
			for (inst = insts.begin() ; inst != insts.end() ; inst++)
			{
				ir::PTXInstruction& ptx = 
					static_cast<ir::PTXInstruction&>(**inst);

				if (ptx.opcode == ir::PTXInstruction::Mov ||
						ptx.opcode == ir::PTXInstruction::Ld ||
						ptx.opcode == ir::PTXInstruction::St)
				{
					ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, 
						&ptx.c};

					for (unsigned int i = 0 ; i != 4 ; i++)
					{
						ir::PTXOperand* operand = operands[i];

						if (operand->addressMode == ir::PTXOperand::Address)
						{
							StringSet::iterator si = 
								external.find(operand->identifier);
							if (si != external.end())
							{
								report("For instruction \""
										<< ptx.toString()
										<< "\", mapping shared label \"" << *si
										<< "\" to external shared memory.");
								externalOperands.push_back(operand);
								continue;
							}

							GlobalMap::iterator gi = 
								sharedGlobals.find(operand->identifier);
							if (gi != sharedGlobals.end())
							{
								ir::Module::GlobalMap::const_iterator it = 
									gi->second;
								sharedGlobals.erase(gi);

								report("Allocating global shared variable \""
										<< it->second.statement.name << "\"");

								map.insert(std::make_pair(
											it->second.statement.name, 
											sharedSize));
								sharedSize += it->second.statement.bytes();
							}

							AllocationMap::iterator mapping = 
								map.find(operand->identifier);
							if (mapping != map.end())
							{
								report("For instruction " << ptx.toString()
										<< ", mapping shared label "
										<< mapping->first << " to " << 
										mapping->second);

								operand->addressMode = 
									ir::PTXOperand::Immediate;
								operand->imm_uint = mapping->second;
							}
						}
					}
				}
			}
		}

		_pad(sharedSize, externalAlignment);

		report("Mapping external shared variables.");
		OperandVector::iterator operand;
		for (operand = externalOperands.begin() ; 
				operand != externalOperands.end() ; operand++)
		{
			report("Mapping external shared label \""
					<< (*operand)->identifier << "\" to " << sharedSize);
			(*operand)->addressMode = ir::PTXOperand::Immediate;
			(*operand)->imm_uint = sharedSize;
		}

		// allocate shared memory object
		_sharedMemorySize = sharedSize;

		report("Total shared memory size is " << _sharedMemorySize);
	}

	void ATIExecutableKernel::_translateKernel()
	{
		report("Translating PTX kernel \"" << name << "\" to IL");

		report("Running IL Translator");
		translator::PTXToILTranslator translator;
		ir::ILKernel *ilKernel = 
			static_cast<ir::ILKernel*>(translator.translate(this));

		report("Assembling il module");
		ilKernel->assemble();

		// query device info
		CalDriver()->calDeviceGetInfo(&_info, 0);

		// compile module
		try {
			CalDriver()->calclCompile(&_object, CAL_LANGUAGE_IL, 
					ilKernel->code().c_str(), _info.target);
		} catch (const hydrazine::Exception& he) {
			std::cerr << "==Ocelot== "
				<< "ATIExecutableKernel failed to compile kernel\n"
				<< std::flush;
			throw;
		}

		// link and load module
		CalDriver()->calclLink(&_image, &_object, 1);
		CalDriver()->calModuleLoad(&_module, *_context, _image);

		delete ilKernel;
	}

	void ATIExecutableKernel::launchGrid(int width, int height, int depth)
	{
		// initialize ABI data
		cb_t *cb0;
		CALuint pitch = 0;
		CALuint flags = 0;

		report("Launching grid");
		report("Grid = " << width << ", " << height);
		report("Block = " << _blockDim.x << ", " << _blockDim.y << ", " 
				<< _blockDim.z);

		cb_t blockDim = {(unsigned int)_blockDim.x,
			(unsigned int)_blockDim.y, (unsigned int)_blockDim.z, 0};
		cb_t gridDim = {(unsigned int)width, (unsigned int)height,
			(unsigned int) depth, 0};

		CalDriver()->calResMap((CALvoid **)&cb0, &pitch, *_cb0Resource, flags);
		cb0[0] = blockDim;
		cb0[1] = gridDim;
		CalDriver()->calResUnmap(*_cb0Resource);

		// translate ptx kernel
		_translateKernel();

		// bind memory handles to module names
		CalDriver()->calCtxGetMem(&_uav0Mem, *_context, *_uav0Resource);
		CalDriver()->calModuleGetName(&_uav0Name, *_context, _module, "uav0");
		CalDriver()->calCtxSetMem(*_context, _uav0Name, _uav0Mem);

		// uav8Name is binded to uav0Mem (for less-than-32bits memory ops)
		CalDriver()->calModuleGetName(&_uav8Name, *_context, _module, "uav8");
		CalDriver()->calCtxSetMem(*_context, _uav8Name, _uav0Mem);

		CalDriver()->calCtxGetMem(&_cb0Mem, *_context, *_cb0Resource);
		CalDriver()->calModuleGetName(&_cb0Name, *_context, _module, "cb0");
		CalDriver()->calCtxSetMem(*_context, _cb0Name, _cb0Mem);

		if (arguments.size()) {
			CalDriver()->calCtxGetMem(&_cb1Mem, *_context, *_cb1Resource);
			CalDriver()->calModuleGetName(&_cb1Name, *_context, _module, "cb1");
			CalDriver()->calCtxSetMem(*_context, _cb1Name, _cb1Mem);
		}

		// get module entry
		CALfunc func = 0;
		CalDriver()->calModuleGetEntry(&func, *_context, _module, "main");

		// invoke kernel
		CALdomain3D gridBlock = {(unsigned int)_blockDim.x,
			(unsigned int)_blockDim.y, (unsigned int)_blockDim.z};
		CALdomain3D gridSize = {(unsigned int)width, (unsigned int)height,
			(unsigned int)depth};

		CALprogramGrid pg;
		pg.func      = func;
		pg.flags     = 0;
		pg.gridBlock = gridBlock;
		pg.gridSize  = gridSize;
		CalDriver()->calCtxRunProgramGrid(_event, *_context, &pg);

		// synchronize
		while(*_event && !CalDriver()->calCtxIsEventDone(*_context, *_event));

		// clean up
		// release memory handles
		CalDriver()->calCtxReleaseMem(*_context, _uav0Mem);
		CalDriver()->calCtxReleaseMem(*_context, _cb0Mem);
		if (arguments.size()) 
			CalDriver()->calCtxReleaseMem(*_context, _cb1Mem);

		// unload module
		CalDriver()->calModuleUnload(*_context, _module);

		// free object and image
		CalDriver()->calclFreeImage(_image);
		CalDriver()->calclFreeObject(_object);
	}

	void ATIExecutableKernel::setKernelShape(int x, int y, int z)
	{
		report("Setting kernel shape: " << x << ", " << y << ", " << z);
		assertM(x * y * z <= 512, "Invalid kernel shape");

		_blockDim.x = x;
		_blockDim.y = y;
		_blockDim.z = z;
	}

	void ATIExecutableKernel::setExternSharedMemorySize(unsigned int bytes)
	{
		report("Setting external shared memory size to " << bytes);
		_externSharedMemorySize = bytes;
	}

	unsigned int ATIExecutableKernel::voteMemorySize() const 
	{ 
		return _voteMemorySize; 
	}

	void ATIExecutableKernel::setVoteMemorySize(unsigned int bytes)
	{
		report("Setting vote memory size to " << bytes);
		_voteMemorySize = bytes;
	}

	void ATIExecutableKernel::setWorkerThreads(unsigned int workerThreadLimit)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIExecutableKernel::updateArgumentMemory()
	{
		report("updateArgumentMemory() - size: " << arguments.size());

		if (arguments.size() == 0) return;

		cb_t *cb1;
		CALuint pitch = 0;
		CALuint flags = 0;

		CalDriver()->calResMap((CALvoid **)&cb1, &pitch, *_cb1Resource, flags);

		int i = 0;
		for (ParameterVector::const_iterator it = arguments.begin();
                it != arguments.end(); it++) {

			report("Updating argument " << it->name <<
					", type " << ir::PTXOperand::toString(it->type) <<
					", array size " << it->arrayValues.size());

			for (unsigned int j = 0 ; j < it->arrayValues.size() ; j++)
			{
				ir::Parameter::ValueType v = it->arrayValues[j];

				switch(it->type) {
					case ir::PTXOperand::u64:
					{
						// CUDA pointers are 32-bits
						assertM(v.val_u64 >> 32 == 0, 
								"Pointer out of range");
						cb1[i].x = v.val_u32; 
						report("cb1[" << i << "] = {" << cb1[i].x << "}");
						i++;
						break;
					}
					case ir::PTXOperand::s8:
					case ir::PTXOperand::s16:
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u8:
					case ir::PTXOperand::u16:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::f16:
					case ir::PTXOperand::f32:
					case ir::PTXOperand::b8:
					case ir::PTXOperand::b16:
					case ir::PTXOperand::b32:
					{
						cb1[i].x = v.val_b32;
						report("cb1[" << i << "] = {" << cb1[i].x << "}");
						i++;
						break;
					}
					default:
					{
						assertM(false, "Parameter type " 
								<< ir::PTXOperand::toString(it->type)
								<< " not supported");
					}
				}
			}
		}

		CalDriver()->calResUnmap(*_cb1Resource);
	}

	void ATIExecutableKernel::initializeGlobalMemory()
	{
		report("Initializing global variables for kernel " << name);

		ir::ControlFlowGraph::iterator block;
		for (block = cfg()->begin() ; block != cfg()->end() ; block++)
		{
			ir::ControlFlowGraph::InstructionList insts = block->instructions;
			ir::ControlFlowGraph::InstructionList::iterator inst;
			for (inst = insts.begin() ; inst != insts.end() ; inst++)
			{
				ir::PTXInstruction& ptx = 
					static_cast<ir::PTXInstruction&>(**inst);

				if (ptx.opcode == ir::PTXInstruction::Mov ||
						ptx.opcode == ir::PTXInstruction::Ld ||
						ptx.opcode == ir::PTXInstruction::St)
				{
					if (ptx.addressSpace != ir::PTXInstruction::Const && 
							ptx.addressSpace != ir::PTXInstruction::Global)
					{
						continue;
					}

					ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, 
						&ptx.c};

					for (unsigned int i = 0 ; i != 4 ; i++)
					{
						ir::PTXOperand* operand = operands[i];

						if (operand->addressMode != ir::PTXOperand::Address)
							continue;

						report("Modifying instruction " << ptx.toString());

						ir::Module::GlobalMap::const_iterator global = 
							module->globals().find(operand->identifier);

						if (global == module->globals().end())
							continue;

						if (device)
						{
							Device::MemoryAllocation* allocation = 
								device->getGlobalAllocation(module->path(),
										global->first);

							operand->addressMode = ir::PTXOperand::Immediate;
							operand->imm_uint = 
								(long long unsigned int)allocation->pointer();
						} else {
							operand->addressMode = ir::PTXOperand::Immediate;
							operand->imm_uint = 0;
						}

						report("Mapping constant label " << global->first 
								<< " to 0x" << std::hex << operand->imm_uint);
					}
				}
			}
		}
	}

	void ATIExecutableKernel::updateGlobals() {
		initializeGlobalMemory();
	}

	void ATIExecutableKernel::updateMemory()
	{
		updateGlobals();
	}

	ExecutableKernel::TextureVector 
		ATIExecutableKernel::textureReferences() const
	{
		assertM(false, "Not implemented yet");
		return ExecutableKernel::TextureVector();
	}

	void ATIExecutableKernel::addTraceGenerator(
		trace::TraceGenerator* generator)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIExecutableKernel::removeTraceGenerator(
		trace::TraceGenerator* generator)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIExecutableKernel::setExternalFunctionSet(
		const ir::ExternalFunctionSet& s)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIExecutableKernel::clearExternalFunctionSet()
	{
		assertM(false, "Not implemented yet");
	}

	inline const cal::CalDriver *ATIExecutableKernel::CalDriver()
	{
		return cal::CalDriver::Instance();
	}

}
