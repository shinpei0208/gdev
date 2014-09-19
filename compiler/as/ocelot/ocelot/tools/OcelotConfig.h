/*! \file OcelotConfig.h
	\author Gregory Diamos
	\date Sunday January 24, 2010
	\brief The header file for the OcelotConfig class.
*/

#ifndef OCELOT_CONFIG_H_INCLUDED
#define OCELOT_CONFIG_H_INCLUDED

#include <string>

namespace util
{
	/*! \brief A class for determining the linker 
		flags required to link against ocelot */
	class OcelotConfig
	{
		private:
			std::string _flags() const;
			std::string _version() const;
			std::string _prefix() const;
			std::string _libs() const;
			std::string _cxxflags() const;
			std::string _includedir() const;
			std::string _libdir() const;
			std::string _bindir() const;
			std::string _tracelibs() const;

		public:
			bool version;
			bool flags;
			bool prefix;
			bool libs;
			bool includedir;
			bool libdir;
			bool bindir;
			bool trace;
	
		public:
			OcelotConfig();
			std::string string() const; 
	};

}

int main(int argc, char** argv);

#endif

