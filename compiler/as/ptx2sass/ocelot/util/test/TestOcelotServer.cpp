/*!
	\file TestOcelotServer.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief tests OcelotServer instance by injecting user-specified messages and waiting for resposne
*/

// C++ includes
#include <string>

// Ocelot includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/util/interface/OcelotServer.h>
#include <ocelot/util/interface/RemoteDeviceMessage.h>

// Boost includes
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>

// hydrazine includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

// whether debugging messages are printed
#define REPORT_BASE 0

/////////////////////////////////////////////////////////////////////////////////////////////////

/*!
	\brief test in which client and server bounce incrementing numbers back and forth 100 times
		then the client disconnects
*/
static void pingpong(boost::asio::ip::tcp::socket &socket, bool verbose) {
	remote::RemoteDeviceMessage message;
	
	int errors = 0;
	for (int n = 0; n < 100 && !errors; n++) {
	
		if (verbose) {
			std::cout << "Iteration " << n << std::flush;
		}
	
		message.clear();
		message.header.operation = remote::RemoteDeviceMessage::Client_ping;
		message.header.messageSize = 4;
		message.resize();
		*((int *)&message.message[0]) = n;
		message.send(socket);
		
		if (verbose) {
			std::cout << " - sent " << std::flush;
		}
		
		message.clear();
		message.receive(socket);
		
		if (verbose) {
			std::cout << " - received " << std::flush;
		}
		
		if (message.header.operation == remote::RemoteDeviceMessage::Client_ping) {
			if (message.size() < 4) {
				std::cout << "FAILED\non iteration " << n << ", server returned message of size " 
					<< message.size() << " instead of size 4" << std::endl;
				++errors;
			}
			else if (*((int *)&message.message[0]) != ~n) {
				std::cout << "FAILED\non iteration " << n << ", server returned " << *((int *)&message.message[0]) <<
					" instead of " << ~n << "\n" << std::endl;
				++errors;
			}
			else {
				if (verbose) {
					std::cout << " - validated " << std::flush;
				}
			}
		}
		else {
			std::cout << "FAILED\non iteration " << n << ", server returned operation " 
				<< remote::RemoteDeviceMessage::toString(message.header.operation) << " instead of ::Client_ping\n" << std::endl;
				++errors;
		}
		if (verbose) {
			std::cout << std::endl;
		}
	}
	std::cout << "Test " << (errors ? "FAILED" : "Passed") << std::endl;
}

int main(int argc, char *argv[]) {
	hydrazine::ArgumentParser parser(argc, argv);
	
	int port;
	std::string host;
	bool verbose;
	
	parser.parse("-p", "--port", port, 2011, "Port to connect on");
	parser.parse("-h", "--host", host, "127.0.0.1", "Remote host to connect to");
	parser.parse("-v", "--verbose", verbose, false, "Verbose printing");
	
	try {
		//using boost::asio::ip::tcp;
    boost::asio::io_service io_service;
    boost::system::error_code error;
    
    boost::asio::ip::tcp::endpoint hostAddress(boost::asio::ip::address::from_string(host), port);

    boost::asio::ip::tcp::socket socket(io_service);
    socket.connect(hostAddress);
    
    if (verbose) {
	    std::cout << "Connecting to: " << host << ":" << port << std::endl;
	   }
		pingpong(socket, verbose);
	}
	catch (std::exception &exp) {
		std::cerr << "TestOcelotServer - " << exp.what() << std::endl;
	}
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

