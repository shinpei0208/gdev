/*!
	\file OcelotServerConnection.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 26, 2011
	\brief connection class for Ocelot server
*/

#ifndef OCELOTSERVERCONNECTION_H_INCLUDED
#define OCELOTSERVERCONNECTION_H_INCLUDED

// Ocelot includes
#include <ocelot/executive/interface/Device.h>
#include <ocelot/util/interface/RemoteDeviceMessage.h>

// Boost includes
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>

// C++ includes
#include <string>
#include <unordered_map>

namespace remote {

	typedef executive::DeviceVector DeviceVector;
	typedef std::unordered_map<std::string, ir::Module*> ModuleMap;

	class OcelotServerConnection {
	public:
		OcelotServerConnection(boost::asio::ip::tcp::socket &socket);
		virtual ~OcelotServerConnection();
		
		//! \brief 
		void start();
		
		void operator()();
		
	private:
	
		//! \brief enumerates enabled devices
		void _enumerateDevices();
		
		//! \brief 
		void _handleMessage(RemoteDeviceMessage &message);
		
		//! \brief Various message handlers
		void _handleAllocate(RemoteDeviceMessage& message);
		void _handleHostToDevice(RemoteDeviceMessage& message);
		void _handleDeviceToHost(RemoteDeviceMessage& message);
		void _handleDeviceToDevice(RemoteDeviceMessage& message);
		void _handleMemset(RemoteDeviceMessage& message);
		void _handleFree(RemoteDeviceMessage& message);
		void _handleClearMemory(RemoteDeviceMessage& message);
		void _handleLoadModule(RemoteDeviceMessage& message);
		void _handleUnloadModule(RemoteDeviceMessage& message);
		void _handleCreateEvent(RemoteDeviceMessage& message);
		void _handleDestroyEvent(RemoteDeviceMessage& message);
		void _handleQueryEvent(RemoteDeviceMessage& message);
		void _handleRecordEvent(RemoteDeviceMessage& message);
		void _handleSynchronizeEvent(RemoteDeviceMessage& message);
		void _handleGetEventTime(RemoteDeviceMessage& message);
		void _handleCreateStream(RemoteDeviceMessage& message);
		void _handleDestroyStream(RemoteDeviceMessage& message);
		void _handleQueryStream(RemoteDeviceMessage& message);
		void _handleSynchStream(RemoteDeviceMessage& message);
		void _handleSetStream(RemoteDeviceMessage& message);
		void _handleBindTexture(RemoteDeviceMessage& message);
		void _handleUnbindTexture(RemoteDeviceMessage& message);
		void _handleDriverVersion(RemoteDeviceMessage& message);
		void _handleRuntimeVersion(RemoteDeviceMessage& message);
		void _handleLaunch(RemoteDeviceMessage& message);
		void _handleGetAttributes(RemoteDeviceMessage& message);
		void _handleGetLastError(RemoteDeviceMessage& message);
		void _handleSynchronize(RemoteDeviceMessage& message);
		void _handleLimitWorkers(RemoteDeviceMessage& message);
		void _handleSetOptimization(RemoteDeviceMessage& message);
		void _handleEnumerateDevices(RemoteDeviceMessage& message);
		void _handlePing(RemoteDeviceMessage& message);
		
		executive::Device& _getDevice(unsigned int id);
		
	private:
		
		//! \brief indicates whether devices have been loaded
		bool _devicesLoaded;
	
		//! \brief list of all devices
		DeviceVector _devices;
		
		//! \brief connection to client
		boost::asio::ip::tcp::socket &_socket;

		//! \brief list of all modules
		ModuleMap _modules;

	};

}

#endif

