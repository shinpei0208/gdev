/*!	\file OcelotServer.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 26, 2011
	\brief implements standalone version of Ocelot for remote devices
*/

#ifndef OCELOTSERVER_H_INCLUDED
#define OCELOTSERVER_H_INCLUDED

// Ocelot Includes
#include <ocelot/util/interface/RemoteDeviceMessage.h>
#include <ocelot/util/interface/OcelotServerConnection.h>

namespace remote {

	class OcelotServer {
	public:
		/*! \brief A vector of created devices */
		typedef std::list< OcelotServerConnection *> OcelotServerConnectionList;
	
	public:
		OcelotServer(unsigned int port);
		~OcelotServer();
		
		//! \brief starts Ocelot in server mode
		void start();
			
	private:
		
		//! \brief handles a message
		void handleMessage(const RemoteDeviceMessage& message);

		//! \brief Various message handlers
		void handleAllocate(const RemoteDeviceMessage& message);
		void handleFree(const RemoteDeviceMessage& message);
		void handleClearMemory(const RemoteDeviceMessage& message);
		void handleLoadModule(const RemoteDeviceMessage& message);
		void handleUnloadModule(const RemoteDeviceMessage& message);
		void handleCreateEvent(const RemoteDeviceMessage& message);
		void handleQueryEvent(const RemoteDeviceMessage& message);
		void handleRecordEvent(const RemoteDeviceMessage& message);
		void handleSynchronizeEvent(const RemoteDeviceMessage& message);
		void handleGetEventTime(const RemoteDeviceMessage& message);
		void handleCreateStream(const RemoteDeviceMessage& message);
		void handleDestroyStream(const RemoteDeviceMessage& message);
		void handleQueryStream(const RemoteDeviceMessage& message);
		void handleSynchStream(const RemoteDeviceMessage& message);
		void handleSetStream(const RemoteDeviceMessage& message);
		void handleBindTexture(const RemoteDeviceMessage& message);
		void handleUnbindTexture(const RemoteDeviceMessage& message);
		void handleDriverVersion(const RemoteDeviceMessage& message);
		void handleRuntimeVersion(const RemoteDeviceMessage& message);
		void handleLaunch(const RemoteDeviceMessage& message);
		void handleGetAttributes(const RemoteDeviceMessage& message);
		void handleGetLastError(const RemoteDeviceMessage& message);
		void handleSynchronize(const RemoteDeviceMessage& message);
		void handleLimitWorkers(const RemoteDeviceMessage& message);
		void handleSetOptimization(const RemoteDeviceMessage& message);
		void handleEnumerateDevices(const RemoteDeviceMessage& message);
	
		//! \brief Send a message back
		void reply(const RemoteDeviceMessage& message);
		
	private:
		OcelotServerConnectionList _connections;
		unsigned int _port;
	};

}

#endif

