/*! \file OcelotConfig.cpp
	\author Gregory Diamos
	\date Sunday January 24, 2010
	\brief The source file for the OcelotConfig class.
*/

#ifndef OCELOT_CONFIG_CPP_INCLUDED
#define OCELOT_CONFIG_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/tools/OcelotConfig.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

// Generated Includes
#include <configure.h>

namespace util
{

	std::string OcelotConfig::_flags() const
	{
		#ifdef OCELOT_CXXFLAGS
		return OCELOT_CXXFLAGS;
		#else
		assertM(false, "Unknown CXX flags, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_version() const
	{
		#ifdef VERSION
		return VERSION;
		#else
		assertM(false, "Unknown version, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_prefix() const
	{
		#ifdef OCELOT_PREFIX_PATH
		return OCELOT_PREFIX_PATH;
		#else
		assertM(false, "Unknown prefix, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_libs() const
	{
		#ifdef OCELOT_LDFLAGS
		return OCELOT_LDFLAGS;
		#else
		assertM(false, "Unknown lib flags, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_includedir() const
	{
		#ifdef OCELOT_INCLUDE_PATH
		return OCELOT_INCLUDE_PATH;
		#else
		assertM(false, "Unknown include dir, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_libdir() const
	{
		#ifdef OCELOT_LIB_PATH
		return OCELOT_LIB_PATH;
		#else
		assertM(false, "Unknown lib dir, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_bindir() const
	{
		#ifdef OCELOT_BIN_PATH
		return OCELOT_BIN_PATH;
		#else
		assertM(false, "Unknown bin dir, is ocelot configured?.");
		#endif
	}

	std::string OcelotConfig::_tracelibs() const
	{
		return "-locelotTrace";
	}

	std::string OcelotConfig::string() const
	{
		std::string result;
		if( version )
		{
			result += _version() + " ";
		}
		if( flags )
		{
			result += _flags() + " ";
		}
		if( prefix )
		{
			result += _prefix() + " ";
		}
		if( libs )
		{
			result += _libs() + " ";
		}
		if( includedir )
		{
			result += _includedir() + " ";
		}
		if( libdir )
		{
			result += _libdir() + " ";
		}
		if( bindir )
		{
			result += _bindir() + " ";
		}

		if( trace )
		{
			result += " " + _tracelibs();
		}
		
		return result + "\n";
	}

	OcelotConfig::OcelotConfig()
	{
	
	}
	
}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	util::OcelotConfig config;
	
	parser.parse( "-l", "--libs", config.libs, false,
		"Libraries needed to link against Ocelot." );
	parser.parse( "-t", "--trace", config.trace, false,
		"Link against ocelot trace generators." );
	parser.parse( "-x", "--cxxflags", config.flags, false,
		"C++ flags for programs that include Ocelot headers." );
	parser.parse( "-L", "--libdir", config.libdir,  false,
		"Directory containing Ocelot libraries." );
	parser.parse( "-i", "--includedir", config.includedir, false,
		"Directory containing Ocelot headers." );
	parser.parse( "-b", "--bindir", config.bindir, false,
		"Directory containing Ocelot executables." );
	parser.parse( "-v", "--version", config.version, false,
		"Print Ocelot version." );
	parser.parse( "-p", "--prefix", config.prefix, false,
		"Print the install prefix." );
	parser.parse();

	std::cout << config.string();
	
	return 0;
}

#endif

