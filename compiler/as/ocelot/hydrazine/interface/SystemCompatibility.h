/*! \file SystemCompatibility.h
	\date Monday August 2, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for hacked code required to assist windows 
		compilaiton
*/

#ifndef SYSTEM_COMPATIBILITY_H_INCLUDED
#define SYSTEM_COMPATIBILITY_H_INCLUDED

// Standard Library Includes
#include <string>
#include <unordered_map>
#include <unordered_set>

/*****************************************************************************\
	Standard Library Includes 
\*****************************************************************************/

namespace hydrazine
{
	/*! \brief Get the number of hardware threads */
	unsigned int getHardwareThreadCount();
	/*! \brief Get the full path to the named executable */
	std::string getExecutablePath(const std::string& executableName);
	/*! \brief The the amount of free physical memory */
	long long unsigned int getFreePhysicalMemory();
	/*! \brief Has there been an OpenGL context bound to this process */
	bool isAnOpenGLContextAvailable();
	/*! \brief Is a string name mangled? */
	bool isMangledCXXString(const std::string& string);
	/*! \brief Demangle a string */
	std::string demangleCXXString(const std::string& string);

}

#endif

