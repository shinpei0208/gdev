/*!
	\file OcelotServer.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 26, 2011
	\brief implements standalone version of Ocelot for remote devices
*/

// Ocelot Includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/tools/OcelotServer.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/Exception.h>

// Boost includes
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>


// Preprocessor Macros
#define Exception(x) { std::stringstream ss; ss << x; \
	throw hydrazine::Exception(ss.str()); }

////////////////////////////////////////////////////////////////////////////////

remote::OcelotServer::OcelotServer(unsigned int p) : _port(p) {

}

remote::OcelotServer::~OcelotServer() {

}

void remote::OcelotServer::start() {
	std::cout << "OcelotServer::start()" << std::endl;
	
	try {
		boost::asio::io_service io_service;
		boost::asio::ip::tcp::acceptor acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), _port));

		while (true) {
			boost::asio::ip::tcp::socket socket(io_service);
			acceptor.accept(socket);
			
			// construct connection thread and push into list of connections
			OcelotServerConnection newConnection(socket);
			
			// right now, single threaded
		}

		io_service.run();
	}
	catch (std::exception& e) {
		std::cerr << "OcelotServer error:\n" << e.what() << std::endl;	
	}
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
	hydrazine::ArgumentParser parser(argc, argv);
	
	std::string configurationFile;
	unsigned int port;

	parser.parse("-c", "--configuration", configurationFile, "configure.ocelot",
		"Name of server configuration file.");
	parser.parse("-p", "--port", port, 2011, "Listening port");
	
	parser.parse();
	
	remote::OcelotServer ocelotServer(port);
	
	ocelotServer.start();
	
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

