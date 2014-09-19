/*!
	\file RemoteDeviceMessage.h
	\author Andrew Kerr <arkerr@gatech.edu>, Gregory Diamos <gregory.diamos@gatech.edu>
	\date 26 Jan 2011
	\brief serialized message object for executive::Device interface
*/

// C++ includes
#include <iostream>

// Ocelot includes
#include <ocelot/util/interface/RemoteDeviceMessage.h>

// Boost includes
#include <boost/array.hpp>

// Hydrazine includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

// whether debugging messages are printed
#define REPORT_BASE 0

/////////////////////////////////////////////////////////////////////////////////////////////////

std::string remote::RemoteDeviceMessage::toString(const Operation &op) {
	switch (op) {
		case Memory_allocate: return "Memory_allocate";
		case Memory_copyHostToDevice: return "Memory_copyHostToDevice";
		case Memory_copyDeviceToHost: return "Memory_copyDeviceToHost";
		case Memory_copyDeviceToDevice: return "Memory_copyDeviceToDevice";
		case Memory_memset: return "Memory_memset";
		case Memory_free: return "Memory_free";
		case Device_clearMemory: return "Device_clearMemory";
		case Device_load: return "Device_load";
		case Device_unload: return "Device_unload";
		case Device_createEvent: return "Device_createEvent";
		case Device_destroyEvent: return "Device_destroyEvent";
		case Device_queryEvent: return "Device_queryEvent";
		case Device_recordEvent: return "Device_recordEvent";
		case Device_synchronizeEvent: return "Device_synchronizeEvent";
		case Device_getEventTime: return "Device_getEventTime";
		case Device_createStream: return "Device_createStream";
		case Device_destroyStream: return "Device_destroyStream";
		case Device_queryStream: return "Device_queryStream";
		case Device_synchronizeStream: return "Device_synchronizeStream";
		case Device_setStream: return "Device_setStream";
		case Device_bindTexture: return "Device_bindTexture";
		case Device_unbindTexture: return "Device_unbindTexture";
		case Device_driverVersion: return "Device_driverVersion";
		case Device_runtimeVersion: return "Device_runtimeVersion";
		case Device_launch: return "Device_launch";
		case Device_getAttributes: return "Device_getAttributes";
		case Device_getLastError: return "Device_getLastError";
		case Device_synchronize: return "Device_synchronize";
		case Device_limitWorkerThreads: return "Device_limitWorkerThreads";
		case Device_setOptimizationLevel: return "Device_setOptimizationLevel";
		case Server_enumerateDevices: return "Server_enumerateDevices";
		
		case Client_acknowledge: return "Client_acknowledge";
		case Client_ping: return "Client_ping";
		case Operation_invalid: return "Operation_invalid";
	}
	return "Operation unknown";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream &operator<<(std::ostream &out, remote::RemoteDeviceMessage::Header header) {
	out << "{ operation: " << remote::RemoteDeviceMessage::toString(header.operation) 
		<< ", deviceId: " << header.deviceId << ", messageSize: " << header.messageSize << " }";
	return out;
}

remote::RemoteDeviceMessage::RemoteDeviceMessage() {
	header.operation = Operation_invalid;
	header.deviceId = -1;
	header.messageSize = 0;
}

remote::RemoteDeviceMessage::~RemoteDeviceMessage() {

}

/*! \brief sends this message on the indicated socket
	\return false if connection disconnected */
bool remote::RemoteDeviceMessage::send(boost::asio::ip::tcp::socket &socket) {
	report("OcelotServerConnection::send() - " << header);
	try {
		size_t len = boost::asio::write( socket, boost::asio::buffer((char *)&header, sizeof(header)));

		assert(len == sizeof(header));
		
		len = boost::asio::write( socket, boost::asio::buffer(message));

	}
	catch (std::exception &exp) {
		std::cerr << "RemoteDeviceMessage::send() - " << exp.what() << std::endl;
	}
	return true;
}

/*! \brief receives this message on indicated socket
	\return false if connection disconnected */
bool remote::RemoteDeviceMessage::receive(boost::asio::ip::tcp::socket &socket) {
	report("OcelotServerConnection::receive()");
	
	try {
		boost::array<char, sizeof(RemoteDeviceMessage::Header)> headerBuffer;
		boost::system::error_code error;

		size_t len = boost::asio::read(socket, boost::asio::buffer(headerBuffer));
		
		std::memcpy(&header, headerBuffer.data(), len);
		
		report("  header: " << header);
		
		resize();
		len = boost::asio::read(socket, boost::asio::buffer(message));
		
	}
	catch (std::exception &exp) {
		std::cerr << "RemoteDeviceMessage::receive() - " << exp.what() << std::endl;
		return false;
	}
	
	report("Received message: " << RemoteDeviceMessage::toString(header.operation) 
		<< " [device " << header.deviceId << "]");
	
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

