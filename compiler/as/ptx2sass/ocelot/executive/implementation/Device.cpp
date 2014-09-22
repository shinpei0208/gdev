/*! \file Device.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Jan 16, 2009
	\brief The source file for the Device class
*/

// Ocelot Includes
#include <ocelot/executive/interface/Device.h>
#include <ocelot/executive/interface/NVIDIAGPUDevice.h>
#include <ocelot/executive/interface/ATIGPUDevice.h>
#include <ocelot/executive/interface/EmulatorDevice.h>
#include <ocelot/executive/interface/MulticoreCPUDevice.h>
#include <ocelot/executive/interface/RemoteDevice.h>
#include <ocelot/executive/interface/PassThroughDevice.h>
#include <ocelot/api/interface/OcelotConfiguration.h>

#include <configure.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

typedef api::OcelotConfiguration config;

executive::Device::MemoryAllocation::MemoryAllocation(bool g, 
	bool h) : _global(g), _host(h)
{

}

executive::Device::MemoryAllocation::~MemoryAllocation()
{

}

bool executive::Device::MemoryAllocation::host() const
{
	return _host;
}

bool executive::Device::MemoryAllocation::global() const
{
	return _global;
}

executive::Device::Properties::Properties(const DeviceProperties& props)
	: DeviceProperties(props)
{
}

std::ostream& executive::Device::Properties::write(std::ostream &out) const
{
	out << name << " ):\n";
	out << "  " << "total memory: " << (totalMemory >> 10) << " kB\n";
	out << "  " << "ISA: " << ir::Instruction::toString(ISA) << "\n";
	out << "  " << "multiprocessors: " << multiprocessorCount << "\n";
	out << "  " << "max threads: " << maxThreadsPerBlock << "\n";
	out << "  " << "shared memory: " << (sharedMemPerBlock >> 10) << " kB\n";
	out << "  " << "const memory: " << (totalConstantMemory >> 10) << " kB\n";
	out << "  " << "SIMD width: " << SIMDWidth << "\n";
	out << "  " << "regs per block: " << regsPerBlock << "\n";
	out << "  " << "clock rate: " << clockRate << " Hz\n";
	return out;
}

executive::DeviceVector executive::Device::createDevices(
	ir::Instruction::Architecture isa, unsigned int flags,
	int computeCapability) 
{
	DeviceVector devices;
	
	switch(isa)
	{
		case ir::Instruction::SASS:
		{
			devices = NVIDIAGPUDevice::createDevices(flags, computeCapability);
		}
		break;
		case ir::Instruction::Emulated:
		{
			devices.push_back(new EmulatorDevice(flags));
		}
		break;
		case ir::Instruction::LLVM:
		{
			#ifdef HAVE_LLVM
			devices.push_back(new MulticoreCPUDevice(flags));
			#endif
		}
		break;
		case ir::Instruction::CAL:
		{
			devices = ATIGPUDevice::createDevices(flags, computeCapability);
		}
		break;
		case ir::Instruction::Remote:
		{
			devices = RemoteDevice::createDevices(flags, computeCapability);
		}
		break;
		default:
		{
			assertM(false, "Invalid ISA - " << ir::Instruction::toString(isa));
		}
	}
	
	if(config::get().checkpoint.enabled) 
	{
		for(DeviceVector::iterator device = devices.begin();
			device != devices.end(); ++device)
		{
			*device = new PassThroughDevice(*device, 0,
				config::get().checkpoint.kernelFilter);
		}
	}
	
	return devices;
}

unsigned int executive::Device::deviceCount(ir::Instruction::Architecture isa,
	int computeCapability) 
{
	switch(isa)
	{
		case ir::Instruction::SASS:
		{
			return NVIDIAGPUDevice::deviceCount(computeCapability);
		}
		break;
		case ir::Instruction::Emulated:
		{
			return 1;
		}
		break;
		case ir::Instruction::LLVM:
		{
			#if HAVE_LLVM
			return 1;
			#else
			return 0;
			#endif
		}
		break;
		case ir::Instruction::CAL:
		{
			return ATIGPUDevice::deviceCount(computeCapability);
		}
		break;
		case ir::Instruction::Remote:
		{
			return RemoteDevice::deviceCount(computeCapability);
		}
		break;
		default: break;
	}
	assertM(false, "Invalid ISA - " << ir::Instruction::toString(isa));
	
	return 0;
}

executive::Device::Device( unsigned int flags) : _driverVersion(4000), 
	_runtimeVersion(4000), _flags(flags) {
	report("Creating device" << this);
}

executive::Device::~Device() {
	report("Destroying device.");
}

bool executive::Device::checkMemoryAccess(const void* pointer, 
	size_t size) const
{
	MemoryAllocation* allocation = getMemoryAllocation(pointer, AnyAllocation);
	if(allocation == 0) return false;
	
	report(" Checking access " << pointer << " (" << size 
		<< " against allocation at " << allocation->pointer() 
		<< " of size " << allocation->size());
	if((char*)pointer + size 
		<= (char*)allocation->pointer() + allocation->size())
	{
		return true;
	}
	
	return false;
}

std::string executive::Device::nearbyAllocationsToString(void* pointer) const
{
	std::stringstream result;
	MemoryAllocationVector allocations = getNearbyAllocations(pointer);
	
	for(MemoryAllocationVector::iterator allocation = allocations.begin(); 
		allocation != allocations.end(); ++allocation)
	{
		result << "[" << (*allocation)->pointer() << "] - [" 
			<< (void*)((char*)(*allocation)->pointer() + (*allocation)->size()) 
			<< "] (" << (*allocation)->size() << " bytes)\n";
	}
	
	return result.str();
}

const executive::Device::Properties& executive::Device::properties() const
{
	return _properties;
}

void executive::Device::select()
{
	boost::thread::id id = boost::this_thread::get_id();

	report("Selecting device for thread " << id << " on device " << this);
	
	_mutex.lock();
	
	ThreadMap::iterator threadState = _selected.find(id);
	
	if(threadState == _selected.end())
	{
		_selected.insert(threadState, std::make_pair(id, true));
	}
	else
	{
		assert(!threadState->second);
			
		threadState->second = true;
	}
	
	_mutex.unlock();
}

bool executive::Device::selected()
{
	bool selected = false;
	
	boost::thread::id id = boost::this_thread::get_id();
	
	report("Is device selected for thread " << id << " on device "
		<< this << " ?");
	
	_mutex.lock();
	
	ThreadMap::const_iterator threadState = _selected.find(id);
	
	if(threadState != _selected.end())
	{
		report(" thread has an entry...");
		selected = threadState->second;
	}
	
	report("  thread was " << (selected ? "selected" : "not selected"));
	
	_mutex.unlock();

	return selected;
}

void executive::Device::unselect()
{
	boost::thread::id id = boost::this_thread::get_id();

	report("Unselecting device for thread " << id << " on device " << this);
	
	_mutex.lock();
	
	ThreadMap::iterator threadState = _selected.find(id);
	
	assert(threadState != _selected.end());

	threadState->second = false;

	_mutex.unlock();
}

int executive::Device::driverVersion() const
{
	return _driverVersion;
}

int executive::Device::runtimeVersion() const
{
	return _runtimeVersion;
}

