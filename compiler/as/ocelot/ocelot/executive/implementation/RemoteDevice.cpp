/*! \file RemoteDevice.h
	\author Andrew Kerr <arkerr@gatech.edu>, Gregory Diamos <gregory.diamos@gatech.edu>
	\date 26 Jan 2011
	\brief class defining a remote Ocelot device 
*/

// Standard library includes
#include <cstring>

// ocelot includes
#include <ocelot/api/interface/OcelotConfiguration.h>
#include <ocelot/executive/interface/RemoteDevice.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// hydrazine includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/Casts.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

// Macros
#define Throw(x) {std::stringstream s; s << x; \
	throw hydrazine::Exception(s.str());}
	
////////////////////////////////////////////////////////////////////////////////

// Turn on report messages
#define REPORT_BASE 0

// Memory Alignment (must be a power of 2) 
// (cuda currently requires 256-byte alignment)
#define ALIGNMENT 256

////////////////////////////////////////////////////////////////////////////////
/*! \brief Allocate a new device for each remote device */
typedef api::OcelotConfiguration config;
typedef remote::RemoteDeviceMessage M;

executive::RemoteDevice::ConnectionManager::ConnectionManager()
	: _socket(_io_service), _connected(false) {

}

executive::RemoteDevice::ConnectionManager::~ConnectionManager() {
	
}

void executive::RemoteDevice::ConnectionManager::_connect() {
	if(_connected) return;

	try {
		//using boost::asio::ip::tcp;
		boost::system::error_code error;

		boost::asio::ip::tcp::endpoint hostAddress(
			boost::asio::ip::address::from_string(config::get().executive.host),
			config::get().executive.port);

		_socket.connect(hostAddress);
		_connected = true;
	}
	catch (const std::exception &exp) {
		report("Connection failed - " << exp.what());
	}
}

void executive::RemoteDevice::ConnectionManager::exchange(
	remote::RemoteDeviceMessage& message)
{
	bool error = false;
	_mutex.lock();
	
	_connect();
	
	try {
		message.send(_socket);
		message.receive(_socket);
	}
	catch (const std::exception &exp) {
		report("Message exchange failed failed - " << exp.what());
		error = true;
	}
	
	_mutex.unlock();
	
	if(error) {
		Throw("Message exchange failed.");
	}
	
	if(message.header.operation 
		!= remote::RemoteDeviceMessage::Client_acknowledge) {
		Throw("Received invalid acknowledgement.");
	}
}

executive::RemoteDevice::ConnectionManager
	executive::RemoteDevice::connectionManager;

executive::Device::DeviceVector executive::RemoteDevice::createDevices(
	unsigned int flags, int computeCapability) {
	if(!config::get().executive.enableRemote) {
		return executive::Device::DeviceVector();
	}
	
	executive::Device::DeviceVector devices;
	
	unsigned int count = 0;

	remote::RemoteDeviceMessage message;

	try {
		message.header.operation
			= remote::RemoteDeviceMessage::Server_enumerateDevices;
		message.header.messageSize = 0;
		message.resize();

		connectionManager.exchange(message);
		
		if(message.size() >= sizeof(unsigned int)) {
			count = *(unsigned int*)message.data(); 
		}
	}
	catch (const std::exception &exp) {
		report("Get device count failed - " << exp.what());
	}
	
	report("Getting " << count << " devices.");
	for(unsigned int i = 0; i < count; ++i) {
		DeviceProperties& props = *(DeviceProperties*)(message.data()
			+ sizeof(unsigned int) + i * sizeof(DeviceProperties));
		devices.push_back(new RemoteDevice(i, props, flags));
	}
	
	return devices;
}

/*! \brief Determine the number of remote GPUs in the system */
unsigned int executive::RemoteDevice::deviceCount(int computeCapability) {
	if(!config::get().executive.enableRemote) {
		return 0;
	}
	
	unsigned int count = 0;

	try {
		remote::RemoteDeviceMessage message;
	
		message.header.operation
			= remote::RemoteDeviceMessage::Server_enumerateDevices;
		message.header.messageSize = 0;
		message.resize();

		connectionManager.exchange(message);
		
		if(message.size() >= sizeof(unsigned int)) {
			count = *(unsigned int*)message.data(); 
		}
	}
	catch (const std::exception &exp) {
		report("Get device count failed - " << exp.what());
	}
	
	report("Found " << count << " remote devices at "
		<< config::get().executive.host << ":" << config::get().executive.port);
	
	return count;
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

/*! \brief Sets the device properties, bind this to the cuda id */
executive::RemoteDevice::RemoteDevice(unsigned int id,
	const DeviceProperties& props, 
	unsigned int flags) : _selected(false), _id(id) {
	_properties = Properties(props);
	
	std::strcat(_properties.name, " (Remote)"); 
}

/*! \brief Clears all state */
executive::RemoteDevice::~RemoteDevice() {
	for(AllocationMap::iterator allocation = _allocations.begin(); 
		allocation != _allocations.end(); ++allocation)
	{
		delete allocation->second;
	}
}

////////////////////////////////////////////////////////////////////////////////

static void* align(void* pointer)
{
	size_t address = (size_t) pointer;
	size_t remainder = address % ALIGNMENT;

	return (void*)(remainder == 0
		? address : address + (ALIGNMENT - remainder));
}

executive::RemoteDevice::MemoryAllocation::MemoryAllocation()
	: Device::MemoryAllocation(false, false), _size(0), _pointer(0),
	_flags(0), _external(false), _device(0) {
}

executive::RemoteDevice::MemoryAllocation::MemoryAllocation(
	RemoteDevice* d, size_t s, void* p)
	: Device::MemoryAllocation(false, false), _size(s), _pointer(p),
	_flags(0), _external(false), _device(d) {
}

executive::RemoteDevice::MemoryAllocation::MemoryAllocation(
	RemoteDevice* d, size_t s,
	unsigned int f)
	: Device::MemoryAllocation(false, true), _size(s),
	_pointer(std::malloc(s + ALIGNMENT)),
	_flags(f), _external(false), _device(d) {
}

executive::RemoteDevice::MemoryAllocation::MemoryAllocation(
	RemoteDevice* d, void* p,
	size_t s)
	: Device::MemoryAllocation(false, false), _size(s), _pointer(p),
	_flags(0), _external(true), _device(d) {
}

executive::RemoteDevice::MemoryAllocation::~MemoryAllocation() {
	if(host()) std::free(_pointer);
}

unsigned int executive::RemoteDevice::MemoryAllocation::flags() const {
	return _flags;
}

void* executive::RemoteDevice::MemoryAllocation::mappedPointer() const {
	assert(host());
	return align(_pointer);
}

void* executive::RemoteDevice::MemoryAllocation::pointer() const {
	assert(!host() || (_flags & cudaHostAllocMapped));
	if(_external) return _pointer;
	return align(_pointer);
}

size_t executive::RemoteDevice::MemoryAllocation::size() const {
	return _size;
}

void executive::RemoteDevice::MemoryAllocation::copy(size_t dOffset,
	const void* host, size_t size) {
	long long unsigned int tSize = 2 * sizeof(long long unsigned int) + size;
	
	report("Memcpy host to device from offset " << dOffset << " of allocation "
		<< pointer() << " of size " << size);
	
	assert(_device != 0);
	remote::RemoteDeviceMessage& _message = _device->_message;
	
	_message.header.operation   = M::Memory_copyHostToDevice;
	_message.header.messageSize = tSize;
	_message.header.deviceId    = _device->_id;
	_message.resize();
	
	unsigned int offset = 0;

	*(long long unsigned int*)(_message.data() + offset) =
		hydrazine::bit_cast<long long unsigned int>(pointer()) + dOffset;
	offset += sizeof(long long unsigned int);

	*(long long unsigned int*)(_message.data() + offset) = size;
	offset += sizeof(long long unsigned int);

	std::memcpy(_message.data() + offset, host, size);	
	offset += size;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

void executive::RemoteDevice::MemoryAllocation::copy(void* host,
	size_t dOffset, size_t size) const {
	long long unsigned int tSize = 2 * sizeof(long long unsigned int);
	
	report("Memcpy device to host from offset " << dOffset << " of allocation "
		<< pointer() << " of size " << size);
	
	assert(_device != 0);
	remote::RemoteDeviceMessage& _message = _device->_message;
	
	_message.header.operation   = M::Memory_copyDeviceToHost;
	_message.header.messageSize = tSize;
	_message.header.deviceId    = _device->_id;
	_message.resize();
	
	unsigned int offset = 0;

	*(long long unsigned int*)(_message.data() + offset) =
		hydrazine::bit_cast<long long unsigned int>(pointer()) + dOffset;
	offset += sizeof(long long unsigned int);

	*(long long unsigned int*)(_message.data() + offset) = size;
	offset += sizeof(long long unsigned int);

	connectionManager.exchange(_message);

	if(_message.size() != size) {
		Throw("Received invalid message from Ocelot server.");
	}

	std::memcpy(host, _message.data(), size);
}

void executive::RemoteDevice::MemoryAllocation::memset(size_t toOffset,
	int value, size_t size) {
	long long unsigned int tSize
		= 2 * sizeof(long long unsigned int) + sizeof(int);
	
	assert(_device != 0);
	remote::RemoteDeviceMessage& _message = _device->_message;
	
	_message.header.operation   = M::Memory_memset;
	_message.header.messageSize = tSize;
	_message.header.deviceId    = _device->_id;
	_message.resize();
	
	unsigned int offset = 0;

	*(long long unsigned int*)(_message.data() + offset) =
		hydrazine::bit_cast<long long unsigned int>(pointer()) + toOffset;
	offset += sizeof(long long unsigned int);

	*(int*)(_message.data() + offset) = value;
	offset += sizeof(int);

	*(long long unsigned int*)(_message.data() + offset) = size;
	offset += sizeof(long long unsigned int);

	connectionManager.exchange(_message);

	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

void executive::RemoteDevice::MemoryAllocation::copy(
	Device::MemoryAllocation* allocation, 
	size_t toOffset, size_t fromOffset, size_t size) const {
	long long unsigned int tSize = 3 * sizeof(long long unsigned int);
	
	report("Memcpy device to device from offset "
		<< fromOffset << " of allocation "
		<< pointer() << " of size " << size);
	
	assert(_device != 0);
	remote::RemoteDeviceMessage& _message = _device->_message;
	
	_message.header.operation   = M::Memory_copyDeviceToDevice;
	_message.header.messageSize = tSize;
	_message.header.deviceId    = _device->_id;
	_message.resize();
	
	unsigned int offset = 0;

	*(long long unsigned int*)(_message.data() + offset) =
		hydrazine::bit_cast<long long unsigned int>(allocation->pointer())
		+ toOffset;
	offset += sizeof(long long unsigned int);

	*(long long unsigned int*)(_message.data() + offset) =
		hydrazine::bit_cast<long long unsigned int>(pointer()) + fromOffset;
	offset += sizeof(long long unsigned int);

	*(long long unsigned int*)(_message.data() + offset) = size;
	offset += sizeof(long long unsigned int);

	connectionManager.exchange(_message);

	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}


executive::Device::MemoryAllocation*
	executive::RemoteDevice::getMemoryAllocation(const void* address, 
	AllocationType type) const {
	assert(selected());

	MemoryAllocation* allocation = 0;
	
	if(type == DeviceAllocation || type == AnyAllocation)
	{
		if(!_allocations.empty())
		{
			AllocationMap::const_iterator alloc = _allocations.upper_bound(
				(void*)address);
			if(alloc != _allocations.begin()) --alloc;
			if(alloc != _allocations.end())
			{
				if(!alloc->second->host()
				 	&& (char*)address >= (char*)alloc->second->pointer())
				{
					allocation = alloc->second;
					return allocation;
				}
			}
		}
	}

	if(type == HostAllocation || type == AnyAllocation)
	{
		for(AllocationMap::const_iterator alloc = _allocations.begin(); 
			alloc != _allocations.end(); ++alloc)
		{
			if(alloc->second->host())
			{
				if((char*)address >= alloc->second->mappedPointer() 
					&& (char*)address < 
					(char*)alloc->second->mappedPointer()
					+ alloc->second->size())
				{
					allocation = alloc->second;
					break;
				}
			}
		}
	}
	
	return allocation;
}

/*! \brief Get the address of a global by stream */
executive::Device::MemoryAllocation*
	executive::RemoteDevice::getGlobalAllocation(
	const std::string& module, const std::string& name) {
	assert(selected());

	assert(0 && "unimplemented");
	return 0;
}

/*! \brief Allocate some new dynamic memory on this device */
executive::Device::MemoryAllocation*
	executive::RemoteDevice::allocate(size_t size) {
	assert(selected());
	
	_message.header.operation   = M::Memory_allocate;
	_message.header.messageSize = sizeof(long long unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(long long unsigned int*)_message.data() = size;
	
	connectionManager.exchange(_message);

	if(_message.size() != sizeof(long long unsigned int)) {
		Throw("Received invalid message from Ocelot server.");
	}

	void* pointer = hydrazine::bit_cast<void*>(
		*(long long unsigned int*)_message.data());

	MemoryAllocation* allocation = new MemoryAllocation(this, size, pointer);
	report("Allocating " << size << " bytes on device at " << pointer
		<< " aligned " << allocation->pointer());
	_allocations.insert(std::make_pair(allocation->pointer(), allocation));
	return allocation;
}

/*! \brief Make this a host memory allocation */
executive::Device::MemoryAllocation* executive::RemoteDevice::allocateHost(
	size_t size, 
	unsigned int flags) {
	assert(selected());

	MemoryAllocation* allocation = new MemoryAllocation(this, size, flags);
	_allocations.insert(std::make_pair(allocation->mappedPointer(), 
		allocation));
	return allocation;
}

/*! \brief Make this a host memory allocation */
executive::Device::MemoryAllocation* executive::RemoteDevice::registerHost(
	void* pointer,
	size_t size, 
	unsigned int flags) {
	assert(selected());

	assert(0 && "unimplemented");
	return 0;
}

/*! \brief Free an existing non-global allocation */
void executive::RemoteDevice::free(void* pointer) {
	assert(selected());
	
	if(pointer == 0) return;
	
	_message.header.operation   = M::Memory_free;
	_message.header.messageSize = sizeof(long long unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(long long unsigned int*)_message.data()
		= hydrazine::bit_cast<long long unsigned int>(pointer);
	
	connectionManager.exchange(_message);

	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}

	AllocationMap::iterator allocation = _allocations.find(pointer);
	if(allocation != _allocations.end())
	{
		if(allocation->second->global())
		{
			Throw("Cannot free global pointer - " << pointer);
		}
		delete allocation->second;
		_allocations.erase(allocation);
		report("Freed pointer " << pointer);
	}
	else
	{
		Throw("Tried to free invalid pointer - " << pointer);
	}
}

/*! \brief Get nearby allocations to a pointer */
executive::Device::MemoryAllocationVector
	executive::RemoteDevice::getNearbyAllocations(void* pointer) const {
	assert(selected());
	MemoryAllocationVector allocations;
	for(AllocationMap::const_iterator allocation = _allocations.begin(); 
		allocation != _allocations.end(); ++allocation)
	{
		allocations.push_back(allocation->second);
	}
	return std::move(allocations);
}

/*! \brief Get all allocations, host, global, and device */
executive::Device::MemoryAllocationVector
	executive::RemoteDevice::getAllAllocations() const {
	assert(selected());
	MemoryAllocationVector allocations;
	for(AllocationMap::const_iterator allocation = _allocations.begin(); 
		allocation != _allocations.end(); ++allocation)
	{
		allocations.push_back(allocation->second);
	}
	
	return allocations;
}

////////////////////////////////////////////////////////////////////////////////

/*! \brief Wipe all memory allocations, but keep modules */
void executive::RemoteDevice::clearMemory() {
	for(AllocationMap::iterator allocation = _allocations.begin(); 
		allocation != _allocations.end();)
	{
		if(allocation->second->global())
		{
			++allocation;			
		}
		else
		{
			delete allocation->second;
			_allocations.erase(allocation++);
		}
	}
	
	_message.header.operation   = M::Device_clearMemory;
	_message.header.messageSize = 0;
	_message.header.deviceId    = _id;
	_message.resize();
	
	connectionManager.exchange(_message);
}

/*! \brief Registers an opengl buffer with a resource */
void* executive::RemoteDevice::glRegisterBuffer(unsigned int buffer, 
	unsigned int flags) {
	assert(selected());

	assert(0 && "unimplemented");
	
	return 0;
}

/*! \brief Registers an opengl image with a resource */
void* executive::RemoteDevice::glRegisterImage(unsigned int image, 
	unsigned int target, unsigned int flags) {
	assert(selected());

	assert(0 && "unimplemented");
	return 0;
}

/*! \brief Unregister a resource */
void executive::RemoteDevice::unRegisterGraphicsResource(void* resource) {
	assert(selected());

	assert(0 && "unimplemented");
}

/*! \brief Map a graphics resource for use with this device */
void executive::RemoteDevice::mapGraphicsResource(void** resource, int count, 
	unsigned int stream) {
	assert(selected());

	assert(0 && "unimplemented");
}

/*! \brief Get a pointer to a mapped resource along with its size */
void* executive::RemoteDevice:: getPointerToMappedGraphicsResource(size_t& size, 
	void* resource) {
	assert(selected());

	assert(0 && "unimplemented");
	return 0;
}

/*! \brief Change the flags of a mapped resource */
void executive::RemoteDevice::setGraphicsResourceFlags(void* resource, 
	unsigned int flags) {
	assert(selected());

	assert(0 && "unimplemented");
};

/*! \brief Unmap a mapped resource */
void executive::RemoteDevice::unmapGraphicsResource(void** resource, int count,
	unsigned int stream) {
	assert(selected());

	assert(0 && "unimplemented");
}


/*! \brief Load a module, must have a unique name */
void executive::RemoteDevice::load(const ir::Module* module) {
	std::stringstream ptx;
	
	module->writeIR(ptx);

	unsigned int size = 2 * sizeof(unsigned int)
		+ ptx.str().size() + module->path().size();
	
	_message.header.operation   = M::Device_load;
	_message.header.messageSize = size;
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data() + offset) = module->path().size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset,
		module->path().data(), module->path().size());	
	offset += module->path().size();

	*(unsigned int*)(_message.data() + offset) = ptx.str().size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, ptx.str().data(), ptx.str().size());	
	offset += ptx.str().size();

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
	
}

/*! \brief Unload a module by name */
void executive::RemoteDevice::unload(const std::string& name) {
	unsigned int size = sizeof(unsigned int) + name.size();
	
	_message.header.operation   = M::Device_unload;
	_message.header.messageSize = size;
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data() + offset) = name.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, name.data(), name.size());	
	offset += name.size();

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Get a translated kernel from the device */
executive::ExecutableKernel* executive::RemoteDevice::getKernel(
	const std::string& module, 
	const std::string& kernel) {
	assert(selected());

	assert(0 && "unimplemented");
	return 0;
}


/*! \brief Create a new event */
unsigned int executive::RemoteDevice::createEvent(int flags) {
	assert(selected());

	_message.header.operation   = M::Device_createEvent;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = flags;

	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(unsigned int)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(unsigned int*)(_message.data());
}

/*! \brief Destroy an existing event */
void executive::RemoteDevice::destroyEvent(unsigned int event) {
	assert(selected());

	_message.header.operation   = M::Device_destroyEvent;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = event;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Query to see if an event has been recorded (yes/no) */
bool executive::RemoteDevice::queryEvent(unsigned int event) {
	assert(selected());

	_message.header.operation   = M::Device_queryEvent;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = event;

	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(bool)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(bool*)(_message.data());
}

/*! \brief Record something happening on an event */
void executive::RemoteDevice::recordEvent(unsigned int event,
	unsigned int stream) {
	assert(selected());

	_message.header.operation   = M::Device_recordEvent;
	_message.header.messageSize = 2 * sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data()) = event;
	offset += sizeof(unsigned int);

	*(unsigned int*)(_message.data() + offset) = stream;
	offset += sizeof(unsigned int);

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Synchronize on an event */
void executive::RemoteDevice::synchronizeEvent(unsigned int event) {
	assert(selected());

	_message.header.operation   = M::Device_synchronizeEvent;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = event;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Get the elapsed time in ms between two recorded events */
float executive::RemoteDevice::getEventTime(unsigned int start,
	unsigned int end) {
	assert(selected());

	_message.header.operation   = M::Device_getEventTime;
	_message.header.messageSize = 2 * sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data()) = start;
	offset += sizeof(unsigned int);

	*(unsigned int*)(_message.data() + offset) = end;
	offset += sizeof(unsigned int);

	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(float)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(float*)(_message.data());
}

////////////////////////////////////////////////////////////////////////////////

/*! \brief Create a new stream */
unsigned int executive::RemoteDevice::createStream() {
	assert(selected());

	_message.header.operation   = M::Device_createStream;
	_message.header.messageSize = 0;
	_message.header.deviceId    = _id;
	_message.resize();
	
	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(unsigned int)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(unsigned int*)(_message.data());
}

/*! \brief Destroy an existing stream */
void executive::RemoteDevice::destroyStream(unsigned int stream) {
	assert(selected());

	_message.header.operation   = M::Device_destroyStream;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = stream;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Query the status of an existing stream (ready/not) */
bool executive::RemoteDevice::queryStream(unsigned int stream) {
	assert(selected());

	_message.header.operation   = M::Device_queryStream;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = stream;

	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(bool)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(bool*)(_message.data());
}

/*! \brief Synchronize a particular stream */
void executive::RemoteDevice::synchronizeStream(unsigned int stream) {
	assert(selected());

	_message.header.operation   = M::Device_synchronizeStream;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = stream;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Sets the current stream */
void executive::RemoteDevice::setStream(unsigned int stream) {
	assert(selected());

	_message.header.operation   = M::Device_setStream;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)(_message.data()) = stream;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

////////////////////////////////////////////////////////////////////////////////

/*! \brief Select this device as the current device. 
	Only one device is allowed to be selected at any time. */
void executive::RemoteDevice::select() {
	assert(!selected());
	_selected = true;
}

/*! \brief is this device selected? */
bool executive::RemoteDevice::selected() const {
	return _selected;
}

/*! \brief Deselect this device. */
void executive::RemoteDevice::unselect() {
	assert(selected());
	_selected = false;
}


////////////////////////////////////////////////////////////////////////////////

/*! \brief Binds a texture to a memory allocation at a pointer */
void executive::RemoteDevice::bindTexture(void* pointer,
	const std::string& module, const std::string& texture, 
	const textureReference& ref, const cudaChannelFormatDesc& desc, 
	const ir::Dim3& dim) {
	assert(selected());
	
	unsigned int size = sizeof(long long unsigned int)
		+ 2 * sizeof(unsigned int)
		+ module.size() + texture.size() + sizeof(textureReference)
		+ sizeof(cudaChannelFormatDesc)
		+ sizeof(ir::Dim3);
	
	_message.header.operation   = M::Device_launch;
	_message.header.messageSize = size;
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(long long unsigned int*)(_message.data() + offset) = module.size();
	offset += sizeof(long long unsigned int);

	*(unsigned int*)(_message.data() + offset) = module.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, module.data(), module.size());	
	offset += module.size();

	*(unsigned int*)(_message.data() + offset) = texture.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, texture.data(), texture.size());	
	offset += texture.size();

	*(textureReference*)(_message.data() + offset) = ref;
	offset += sizeof(textureReference);

	*(cudaChannelFormatDesc*)(_message.data() + offset) = desc;
	offset += sizeof(cudaChannelFormatDesc);

	*(ir::Dim3*)(_message.data() + offset) = dim;
	offset += sizeof(ir::Dim3);

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief unbinds anything bound to a particular texture */
void executive::RemoteDevice::unbindTexture(const std::string& module, 
	const std::string& texture) {
	assert(selected());

	unsigned int size = 2 * sizeof(unsigned int)
		+ module.size() + texture.size();
	
	_message.header.operation   = M::Device_getAttributes;
	_message.header.messageSize = size;
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data() + offset) = texture.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, module.data(), module.size());	
	offset += module.size();

	*(unsigned int*)(_message.data() + offset) = texture.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, texture.data(), texture.size());	
	offset += texture.size();

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}

/*! \brief Get's a reference to an internal texture */
void* executive::RemoteDevice::getTextureReference(
	const std::string& moduleName, 
	const std::string& textureName) {
	assert(selected());

	assert(0 && "unimplemented");
	return 0;
}


////////////////////////////////////////////////////////////////////////////////

/*! \brief helper function for launching a kernel
	\param module module name
	\param kernel kernel name
	\param grid grid dimensions
	\param block block dimensions
	\param sharedMemory shared memory size
	\param argumentBlock array of bytes for parameter memory
	\param argumentBlockSize number of bytes in parameter memory
	\param traceGenerators vector of trace generators to add 
		and remove from kernel
*/
void executive::RemoteDevice::launch(const std::string& module, 
	const std::string& kernel, const ir::Dim3& grid, 
	const ir::Dim3& block, size_t sharedMemory, 
	const void* argumentBlock, size_t argumentBlockSize, 
	const trace::TraceGeneratorVector & traceGenerators,
	const ir::ExternalFunctionSet* externals) {
	assert(selected());

	unsigned int size = 4 * sizeof(unsigned int)
		+ module.size() + kernel.size() + argumentBlockSize
		+ 2 * sizeof(ir::Dim3);
	
	_message.header.operation   = M::Device_launch;
	_message.header.messageSize = size;
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data() + offset) = module.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, module.data(), module.size());	
	offset += module.size();

	*(unsigned int*)(_message.data() + offset) = kernel.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, kernel.data(), kernel.size());	
	offset += kernel.size();

	*(ir::Dim3*)(_message.data() + offset) = grid;
	offset += sizeof(ir::Dim3);

	*(ir::Dim3*)(_message.data() + offset) = block;
	offset += sizeof(ir::Dim3);

	*(unsigned int*)(_message.data() + offset) = sharedMemory;
	offset += sizeof(unsigned int);

	*(unsigned int*)(_message.data() + offset) = argumentBlockSize;
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, argumentBlock, argumentBlockSize);	
	offset += argumentBlockSize;

	connectionManager.exchange(_message);
	
	if(_message.size() != 0) {
		Throw("Received invalid message from Ocelot server.");
	}
}


////////////////////////////////////////////////////////////////////////////////

/*! \brief Get the function attributes of a specific kernel */
cudaFuncAttributes executive::RemoteDevice::getAttributes(
	const std::string& module, 
	const std::string& kernel) {
	assert(selected());
	
	unsigned int size = 2 * sizeof(unsigned int)
		+ module.size() + kernel.size();
	
	_message.header.operation   = M::Device_getAttributes;
	_message.header.messageSize = size;
	_message.header.deviceId    = _id;
	_message.resize();
	
	unsigned int offset = 0;

	*(unsigned int*)(_message.data() + offset) = module.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, module.data(), module.size());	
	offset += module.size();

	*(unsigned int*)(_message.data() + offset) = kernel.size();
	offset += sizeof(unsigned int);

	std::memcpy(_message.data() + offset, kernel.data(), kernel.size());	
	offset += kernel.size();

	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(cudaFuncAttributes)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(cudaFuncAttributes*)_message.data();
}

/*! \brief Get the last error from this device */
unsigned int executive::RemoteDevice::getLastError() {
	assert(selected());
	
	_message.header.operation   = M::Device_getLastError;
	_message.header.messageSize = 0;
	_message.header.deviceId    = _id;
	_message.resize();
	
	connectionManager.exchange(_message);
	
	if(_message.size() != sizeof(unsigned int)) {
		Throw("Received invalid message from Ocelot server.");
	}
	
	return *(unsigned int*)_message.data();
}

/*! \brief Wait until all asynchronous operations have completed */
void executive::RemoteDevice::synchronize() {
	assert(selected());
	
	_message.header.operation   = M::Device_synchronize;
	_message.header.messageSize = 0;
	_message.header.deviceId    = _id;
	_message.resize();
	
	connectionManager.exchange(_message);
}

/*! \brief Limit the worker threads used by this device */
void executive::RemoteDevice::limitWorkerThreads(unsigned int threads) {
	assert(selected());
	
	_message.header.operation   = M::Device_limitWorkerThreads;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)_message.data() = threads;
	
	connectionManager.exchange(_message);
}
			
/*! \brief Set the optimization level for kernels in this device */
void executive::RemoteDevice::setOptimizationLevel(
	translator::Translator::OptimizationLevel level) {
	assert(selected());

	_message.header.operation   = M::Device_setOptimizationLevel;
	_message.header.messageSize = sizeof(unsigned int);
	_message.header.deviceId    = _id;
	_message.resize();
	
	*(unsigned int*)_message.data() = level;
	
	connectionManager.exchange(_message);
}

////////////////////////////////////////////////////////////////////////////////

