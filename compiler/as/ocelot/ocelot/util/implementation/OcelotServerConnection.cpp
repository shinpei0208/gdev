/*!
	\file OcelotServerConnection.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 26, 2011
	\brief connection class for Ocelot server
*/

// C++ includes
#include <iostream>


// Ocelot includes
#include <ocelot/util/interface/OcelotServerConnection.h>
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/api/interface/OcelotConfiguration.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/executive/interface/RuntimeException.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// Hydrazine includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Casts.h>


#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////

// whether CUDA runtime catches exceptions thrown by Ocelot
#define CATCH_RUNTIME_EXCEPTIONS 0

// whether verbose error messages are printed
#define CUDA_VERBOSE 0

// whether debugging messages are printed
#define REPORT_BASE 1

// report all ptx modules
#define REPORT_ALL_PTX 0

// Exception shorthand
#define Exception(x) { std::stringstream ss; ss << x; \
	throw hydrazine::Exception(ss.str()); }

/////////////////////////////////////////////////////////////////////////////////////////////////

static std::string readString(const void* data, unsigned int offset)
{
	const char* buffer = (const char*) data + offset;
	
	unsigned int size = *(unsigned int*)buffer;
	buffer += sizeof(unsigned int);
	
	std::string result(size, ' ');
	
	std::memcpy((char*)result.data(), buffer, size);

	return result;
}

typedef api::OcelotConfiguration config;

remote::OcelotServerConnection::OcelotServerConnection(boost::asio::ip::tcp::socket &socket)
:
	_devicesLoaded(false),
	_socket(socket)
{
	report("OcelotServerConnection()");
	start();
}

remote::OcelotServerConnection::~OcelotServerConnection() {
	//
	// free things that need freeing
	//
	// devices
	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		delete *device;
	}
	
	// mutex

	// thread contexts
	
	// textures
	
	// kernels
	
	// modules
	for (ModuleMap::iterator module = _modules.begin(); 
		module != _modules.end(); ++module) {
		delete module->second;
	}
	// config
	config::destroy();
}

void remote::OcelotServerConnection::operator()() {
	start();
}

//! \brief 
void remote::OcelotServerConnection::start() {

	report("OcelotServerConnection::start()");
	
	_enumerateDevices();

	RemoteDeviceMessage message;
	bool running = true;
	while (running) {
		// clear message
		message.clear();
		
		// receive		
		running = message.receive(_socket);
		
		report("Server received message - " << message.header);
		
		// dispatch
		try {
			_handleMessage(message);
		}
		catch(const std::exception& e) {
			std::cerr << "Error in message handler - " << e.what() << "\n";
			try {
				message.clear();
				message.header.operation = RemoteDeviceMessage::Operation_invalid;
				message.send(_socket);
			}
			catch (const std::exception& e) {
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////


//! 
void remote::OcelotServerConnection::_enumerateDevices() {

	report("Creating devices.");
	
	if(_devicesLoaded) return;
	
	int _flags = 0;
	int _computeCapability = 2;
	
	if(config::get().executive.enableNVIDIA) {
		executive::DeviceVector d = 
			executive::Device::createDevices(ir::Instruction::SASS, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " nvidia gpu devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	if(config::get().executive.enableEmulated) {
		executive::DeviceVector d = 
			executive::Device::createDevices(ir::Instruction::Emulated, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " emulator devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	if(config::get().executive.enableLLVM) {
		executive::DeviceVector d = 
			executive::Device::createDevices(ir::Instruction::LLVM, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " llvm-cpu devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
		
		if (config::get().executive.workerThreadLimit > 0) {
			for (executive::DeviceVector::iterator d_it = d.begin();
				d_it != d.end(); ++d_it) {
				(*d_it)->limitWorkerThreads(
					config::get().executive.workerThreadLimit);
			}
		}
	}
	if(config::get().executive.enableAMD) {
		executive::DeviceVector d =
			executive::Device::createDevices(ir::Instruction::CAL, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " amd gpu devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	
	// Remote devices
	{
	
	}

	// done enumerating	
	_devicesLoaded = true;
	
	if(_devices.empty())
	{
		std::cerr << "==Ocelot== WARNING - No CUDA devices found or all " 
			<< "devices disabled!\n";
		std::cerr << "==Ocelot==  Consider enabling the emulator in " 
			<< "configure.ocelot.\n";
	}
}

void remote::OcelotServerConnection::_handleMessage(RemoteDeviceMessage& message) {
	typedef RemoteDeviceMessage M;
	
	if(message.header.operation == M::Server_enumerateDevices) {
		_handleEnumerateDevices(message);
		return;
	}
	
	if(message.header.operation == M::Client_ping) {
		_handlePing(message);
		return;
	}
	executive::Device& device = _getDevice(message.header.deviceId);
		
	device.select();
	
	switch(message.header.operation) {
	case M::Memory_allocate:             _handleAllocate(message); break;
	case M::Memory_copyHostToDevice:     _handleHostToDevice(message); break;
	case M::Memory_copyDeviceToHost:     _handleDeviceToHost(message); break;
	case M::Memory_copyDeviceToDevice:   _handleDeviceToDevice(message); break;
	case M::Memory_memset:               _handleMemset(message); break;
	case M::Memory_free:                 _handleFree(message); break;
	case M::Device_clearMemory:          _handleClearMemory(message); break;
	case M::Device_load:                 _handleLoadModule(message); break;
	case M::Device_unload:               _handleUnloadModule(message); break;
	case M::Device_createEvent:          _handleCreateEvent(message); break;
	case M::Device_destroyEvent:         _handleDestroyEvent(message); break;
	case M::Device_queryEvent:           _handleQueryEvent(message); break;
	case M::Device_recordEvent:          _handleRecordEvent(message); break;
	case M::Device_synchronizeEvent:     _handleSynchronizeEvent(message); break;
	case M::Device_getEventTime:         _handleGetEventTime(message); break;
	case M::Device_createStream:         _handleCreateStream(message); break;
	case M::Device_destroyStream:        _handleDestroyStream(message); break;
	case M::Device_queryStream:          _handleQueryStream(message); break;
	case M::Device_synchronizeStream:    _handleSynchStream(message); break;
	case M::Device_setStream:            _handleSetStream(message); break;
	case M::Device_bindTexture:          _handleBindTexture(message); break;
	case M::Device_unbindTexture:        _handleUnbindTexture(message); break;
	case M::Device_driverVersion:        _handleDriverVersion(message); break;
	case M::Device_runtimeVersion:       _handleRuntimeVersion(message); break;
	case M::Device_launch:               _handleLaunch(message); break;
	case M::Device_getAttributes:        _handleGetAttributes(message); break;
	case M::Device_getLastError:         _handleGetLastError(message); break;
	case M::Device_synchronize:          _handleSynchronize(message); break;
	case M::Device_limitWorkerThreads:   _handleLimitWorkers(message); break;
	case M::Device_setOptimizationLevel: _handleSetOptimization(message); break;
	default: Exception("Invalid operation code."); 
	}
	
	device.unselect();
}

// Message format is (size), ack is (pointer)
void remote::OcelotServerConnection::_handleAllocate(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(long long unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_allocate message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	long long unsigned int size = *(long long unsigned int*)message.data();

	executive::Device::MemoryAllocation* allocation = device.allocate(size);
	
	*(long long unsigned int*)message.data()
		= hydrazine::bit_cast<long long unsigned int>(allocation->pointer());

	report("Allocating device region of size " << size
		<< " at " << allocation->pointer());

	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (pointer, string), ack is blank
void remote::OcelotServerConnection::_handleHostToDevice(RemoteDeviceMessage& message) {
	if(message.size() < sizeof(long long unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_copyHostToDevice message.");
	}

	executive::Device& device = _getDevice(message.header.deviceId);
	
	long long unsigned int pointer = *(long long unsigned int*)message.data();
	long long unsigned int size    = *(long long unsigned int*)(message.data()
		+ sizeof(long long unsigned int));

	executive::Device::MemoryAllocation* allocation = 
		device.getMemoryAllocation(hydrazine::bit_cast<void*>(pointer));
	size_t offset = (char*)hydrazine::bit_cast<char*>(pointer)
		- (char*)allocation->pointer();

	report("Memcpy host to device from offset " << offset << " of allocation "
		<< allocation->pointer() << " of size " << size);

	allocation->copy(offset,
		message.data() + 2 * sizeof(long long unsigned int), size);
	
	message.header.messageSize = 0;
	message.resize();	
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (pointer, size), ack is (string)
void remote::OcelotServerConnection::_handleDeviceToHost(RemoteDeviceMessage& message) {
	if(message.size() != 2 * sizeof(long long unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_copyDeviceToHost message.");
	}

	executive::Device& device = _getDevice(message.header.deviceId);
	
	long long unsigned int pointer = *(long long unsigned int*)message.data();
	long long unsigned int size    = *(long long unsigned int*)(message.data()
		+ sizeof(long long unsigned int));

	executive::Device::MemoryAllocation* allocation = 
		device.getMemoryAllocation(hydrazine::bit_cast<void*>(pointer));
	size_t offset = (char*)hydrazine::bit_cast<char*>(pointer)
		- (char*)allocation->pointer();

	report("Memcpy device to host from offset " << offset << " of allocation "
		<< allocation->pointer() << " of size " << size);

	message.header.messageSize = size;
	message.resize();	

	allocation->copy(message.data(), offset, size);
		
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (pointer, pointer, size), ack is blank
void remote::OcelotServerConnection::_handleDeviceToDevice(RemoteDeviceMessage& message) {
	if(message.size() != 3 * sizeof(long long unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_copyDeviceToDevice message.");
	}

	executive::Device& device = _getDevice(message.header.deviceId);
	
	long long unsigned int pointerD = *(long long unsigned int*)message.data();
	long long unsigned int pointerS = *(long long unsigned int*)message.data();
	long long unsigned int size     = *(long long unsigned int*)(message.data()
		+ 2 * sizeof(long long unsigned int));

	executive::Device::MemoryAllocation* allocationD = 
		device.getMemoryAllocation(hydrazine::bit_cast<void*>(pointerD));
	size_t offsetD = (char*)hydrazine::bit_cast<char*>(pointerD)
		- (char*)allocationD->pointer();

	executive::Device::MemoryAllocation* allocationS = 
		device.getMemoryAllocation(hydrazine::bit_cast<void*>(pointerS));
	size_t offsetS = (char*)hydrazine::bit_cast<char*>(pointerS)
		- (char*)allocationS->pointer();

	report("Memcpy device to host from offset " << offsetS << " of allocation "
		<< allocationS->pointer() << " of size " << size << " to offset "
		<< offsetD << " of allocation " << allocationD->pointer());

	message.header.messageSize = 0;
	message.resize();	

	allocationS->copy(allocationD, offsetD, offsetS, size);
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (pointer, int, size), ack is blank
void remote::OcelotServerConnection::_handleMemset(RemoteDeviceMessage& message) {
	if(message.size() != 2 * sizeof(long long unsigned int) + sizeof(int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_copyDeviceToDevice message.");
	}

	executive::Device& device = _getDevice(message.header.deviceId);
	
	long long unsigned int pointer = *(long long unsigned int*)message.data();
	int value    = *(int*)(message.data() + sizeof(long long unsigned int));
	long long unsigned int size    = *(long long unsigned int*)(message.data()
		+ sizeof(long long unsigned int) + sizeof(unsigned int));

	executive::Device::MemoryAllocation* allocation = 
		device.getMemoryAllocation(hydrazine::bit_cast<void*>(pointer));
	size_t offset = (char*)hydrazine::bit_cast<char*>(pointer)
		- (char*)allocation->pointer();

	report("Memset at offset " << offset << " of allocation "
		<< allocation->pointer() << " of size " << size);

	message.header.messageSize = 0;
	message.resize();	

	allocation->memset(offset, value, size);
		
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (pointer), ack is blank
void remote::OcelotServerConnection::_handleFree(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(long long unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_allocate message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	long long unsigned int pointer = *(long long unsigned int*)message.data();

	device.free(hydrazine::bit_cast<void*>(pointer));
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is blank
void remote::OcelotServerConnection::_handleClearMemory(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for Memory_allocate message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.clearMemory();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (string, string), ack is blank
void remote::OcelotServerConnection::_handleLoadModule(RemoteDeviceMessage& message) {
	std::string name = readString(message.data(), 0);

	ModuleMap::iterator module = _modules.find(name);
	if(module == _modules.end()) {
	
		std::string ptx  = readString(message.data(),
			sizeof(unsigned int) + name.size());
	
		std::stringstream ptxstream(ptx);
	
		module = _modules.insert(std::make_pair(name,
			new ir::Module(ptxstream, name))).first;
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	message.header.messageSize = 0;
	message.resize();

	device.load(module->second);

	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (string), ack is blank
void remote::OcelotServerConnection::_handleUnloadModule(RemoteDeviceMessage& message) {
	std::string module = readString(message.data(), 0);
	
	message.header.messageSize = 0;
	message.resize();
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.unload(module);
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is (unsigned int)
void remote::OcelotServerConnection::_handleCreateEvent(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for CreateEvent message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	unsigned int result = device.createStream();
	
	message.header.messageSize = sizeof(unsigned int);
	message.resize();
	
	*(unsigned int*)message.data() = result;
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleDestroyEvent(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for CreateEvent message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.destroyEvent(*(unsigned int*)message.data());
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}
// Message format is (unsigned int), ack is (bool)
void remote::OcelotServerConnection::_handleQueryEvent(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for QueryEvent message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	bool result = device.queryEvent(*(unsigned int*)message.data());
	
	message.header.messageSize = sizeof(bool);
	message.resize();
	
	*(bool*)message.data() = result;
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int, unsigned int), ack is blank
void remote::OcelotServerConnection::_handleRecordEvent(RemoteDeviceMessage& message) {
	if(message.size() != 2 * sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for RecordEvent message.");
	}
	
	unsigned int event  = *(unsigned int*)message.data();
	unsigned int stream =
		*(unsigned int*)(message.data() + sizeof(unsigned int));
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.recordEvent(event, stream);
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleSynchronizeEvent(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for SynchronizeStream message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.synchronizeStream(*(unsigned int*)message.data());
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int, unsigned int), ack is (float)
void remote::OcelotServerConnection::_handleGetEventTime(RemoteDeviceMessage& message) {
	if(message.size() != 2 * sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for GetEventTime message.");
	}
	
	unsigned int start  = *(unsigned int*)message.data();
	unsigned int finish =
		*(unsigned int*)(message.data() + sizeof(unsigned int));
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	float time = device.getEventTime(start, finish);
	
	message.header.messageSize = sizeof(float);
	message.resize();
	
	*(float*)message.data() = time;
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is (unsigned int)
void remote::OcelotServerConnection::_handleCreateStream(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for CreateStream message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	unsigned int result = device.createStream();
	
	message.header.messageSize = sizeof(unsigned int);
	message.resize();
	
	*(unsigned int*)message.data() = result;
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleDestroyStream(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for DestroyStream message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.destroyStream(*(unsigned int*)message.data());
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is (bool)
void remote::OcelotServerConnection::_handleQueryStream(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for SynchronizeStream message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	bool result = device.queryStream(*(unsigned int*)message.data());
	
	message.header.messageSize = sizeof(bool);
	message.resize();
	
	*(bool*)message.data() = result;
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleSynchStream(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for SynchronizeStream message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.synchronizeStream(*(unsigned int*)message.data());
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleSetStream(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for SetStream message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.setStream(*(unsigned int*)message.data());
	
	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (pointer, string, string, texref, desc, dim3), ack is blank
void remote::OcelotServerConnection::_handleBindTexture(RemoteDeviceMessage& message) {
	unsigned int offset = 0;

	long long unsigned int pointer
		= *(long long unsigned int*)(message.data() + offset);
	offset += sizeof(long long unsigned int);
	
	std::string module = readString(message.data(), offset);
	offset += sizeof(unsigned int) + module.size();
	
	std::string name   = readString(message.data(), offset);
	offset += sizeof(unsigned int) + name.size();
	
	textureReference ref = *(textureReference*)(message.data() + offset);
	offset += sizeof(textureReference);
	
	cudaChannelFormatDesc desc =
		*(cudaChannelFormatDesc*)(message.data() + offset);
	offset += sizeof(cudaChannelFormatDesc);
	
	ir::Dim3 size = *(ir::Dim3*)(message.data() + offset);
	offset += sizeof(ir::Dim3);
	
	message.header.messageSize = 0;
	message.resize();
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.bindTexture(hydrazine::bit_cast<void*>(pointer), module,
		name, ref, desc, size);
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (string, string), ack is blank
void remote::OcelotServerConnection::_handleUnbindTexture(RemoteDeviceMessage& message) {
	std::string module = readString(message.data(), 0);
	std::string kernel = readString(message.data(),
		sizeof(unsigned int) + module.size());
	
	message.header.messageSize = 0;
	message.resize();
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.unbindTexture(module, kernel);
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is (unsigned int)
void remote::OcelotServerConnection::_handleDriverVersion(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for GetDriverVersion message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	message.header.messageSize = sizeof(unsigned int);
	message.resize();
	
	*(unsigned int*)message.data() = device.driverVersion();

	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is (unsigned int)
void remote::OcelotServerConnection::_handleRuntimeVersion(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for GetRuntimeVersion message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	message.header.messageSize = sizeof(unsigned int);
	message.resize();
	
	*(unsigned int*)message.data() = device.runtimeVersion();

	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (string, string, dim3, dim3, int, string), ack is blank
void remote::OcelotServerConnection::_handleLaunch(RemoteDeviceMessage& message) {
	unsigned int offset = 0;
	std::string moduleName = readString(message.data(), 0);
	offset += sizeof(unsigned int) + moduleName.size();

	std::string kernelName = readString(message.data(), offset);
	offset += sizeof(unsigned int) + kernelName.size();
	
	ir::Dim3 grid = *(ir::Dim3*)(message.data() + offset);
	offset += sizeof(ir::Dim3);
	
	ir::Dim3 block = *(ir::Dim3*)(message.data() + offset);
	offset += sizeof(ir::Dim3);
	
	unsigned int sharedSize = *(unsigned int*)(message.data() + offset);
	offset += sizeof(unsigned int);
	
	std::string arguments = readString(message.data(), offset);
	
	message.header.messageSize = 0;
	message.resize();
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.launch(moduleName, kernelName, grid, block, sharedSize,
		arguments.c_str(), arguments.size());
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (string, string), ack is (attributes)
void remote::OcelotServerConnection::_handleGetAttributes(RemoteDeviceMessage& message) {
	std::string moduleName = readString(message.data(), 0);
	std::string kernelName = readString(message.data(),
		sizeof(unsigned int) + moduleName.size());

	executive::Device& device = _getDevice(message.header.deviceId);
	
	message.header.messageSize = sizeof(cudaFuncAttributes);
	message.resize();
	
	*(cudaFuncAttributes*)message.data() = device.getAttributes(
		moduleName, kernelName);

	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is (unsigned int)
void remote::OcelotServerConnection::_handleGetLastError(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for GetLastError message.");
	}
	
	executive::Device& device = _getDevice(message.header.deviceId);
	
	message.header.messageSize = sizeof(unsigned int);
	message.resize();
	
	*(unsigned int*)message.data() = device.getLastError();

	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is blank
void remote::OcelotServerConnection::_handleSynchronize(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for Synchronize message.");
	}
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.synchronize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleLimitWorkers(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for LimitWorkerThreads message.");
	}
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.limitWorkerThreads(*(unsigned int*)message.data());

	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is (unsigned int), ack is blank
void remote::OcelotServerConnection::_handleSetOptimization(RemoteDeviceMessage& message) {
	if(message.size() != sizeof(unsigned int)) {
		Exception("Invalid data size '" << message.size()
			<< "' for SetOptimizationLevel message.");
	}
	executive::Device& device = _getDevice(message.header.deviceId);
	
	device.setOptimizationLevel((translator::Translator::OptimizationLevel)
		*(unsigned int*)message.data());

	message.header.messageSize = 0;
	message.resize();
	
	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

// Message format is blank, ack is (unsigned int)
void remote::OcelotServerConnection::_handleEnumerateDevices(RemoteDeviceMessage& message) {
	if(message.size() != 0) {
		Exception("Invalid data size '" << message.size()
			<< "' for EnumerateDevices message.");
	}
	
	message.header.messageSize = sizeof(unsigned int)
		+ _devices.size() * sizeof(executive::DeviceProperties);
	message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)message.data() = _devices.size();
	offset += sizeof(unsigned int);
	
	for(unsigned int i = 0; i < _devices.size(); ++i)
	{
		std::memcpy((message.data() + offset), &_devices[i]->properties(),
			sizeof(executive::DeviceProperties));
		offset += sizeof(executive::DeviceProperties);
	}

	message.header.operation = RemoteDeviceMessage::Client_acknowledge;
	message.send(_socket);
}

executive::Device& remote::OcelotServerConnection::_getDevice(unsigned int id) {
	if(id >= _devices.size()) {
		Exception("Device id " << id << " is out of range.");
	}
	
	return *_devices[id];
}

void remote::OcelotServerConnection::_handlePing(RemoteDeviceMessage& message) {
    if (message.size() < sizeof(int)) {
    	Exception("Invalid ping message.");
    }
    
    *((int *)&message.message[0]) = ~(*((int *)&message.message[0]));
    message.send(_socket);
}
/////////////////////////////////////////////////////////////////////////////////////////////////

