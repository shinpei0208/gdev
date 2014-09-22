/*! \file   compression.cpp
	\date   Wednesday December 5, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file the hydrazine compression abstraction
*/

// Hydrazine Includes
#include <hydrazine/interface/compression.h>
#include <hydrazine/interface/Casts.h>

// System Includes
#if __GNUC__
	#include <dlfcn.h>
#else 
	// TODO Add dynamic loading support on windows
	#define dlopen(a,b) 0
	#define dlclose(a) -1
	#define dlerror() "Unknown error"
	#define dlsym(a,b) 0
#endif

// Standard Library Includes
#include <stdexcept>
#include <cstdint>

namespace hydrazine
{

void decompress(void* output, uint64_t& outputSize, const void* input,
	uint64_t inputSize)
{
	#if __GNUC__
	void* libz = dlopen("libz.so", RTLD_LAZY);
	
	if(libz == 0)
	{
		throw std::runtime_error(
			"Failed to load compression library 'libz.so'");
	}
	
	typedef int (*UncompressFunction)(uint8_t*, unsigned long*, const uint8_t*,
		unsigned long);
	
	UncompressFunction uncompress = hydrazine::bit_cast<UncompressFunction>(
		dlsym(libz, "uncompress"));
	
	int result = uncompress((uint8_t*)output, (unsigned long*)&outputSize,
		(const uint8_t*)input, (unsigned long) inputSize);
	
	dlclose(libz);
	
	if(result != 0)
	{
		std::stringstream errorcode;
		
		errorcode << result;
	
		throw std::runtime_error("Failed to decompress input. ("
			+ errorcode.str() + ")");
	}
	
	#else
	
	throw std::runtime_error("Compression not supported on this platform.");
	
	#endif
}

}


