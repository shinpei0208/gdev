/*! \file OcelotRuntime.cpp
	\author Gregory Diamos
	\date Tuesday August 11, 2009
	\brief The source file for the OcelotRuntime class.
*/

#ifndef OCELOT_RUNTIME_CPP_INCLUDED
#define OCELOT_RUNTIME_CPP_INCLUDED

// Ocelot includes
#include <ocelot/api/interface/OcelotRuntime.h>
#include <ocelot/api/interface/ocelot.h>

#include <ocelot/cuda/interface/cuda_runtime.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Casts.h>

// Confguration
#include <configure.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

static void* cudaMallocWrapper(size_t bytes)
{
	void* pointer = 0;
	
	#if EXCLUDE_CUDA_RUNTIME == 0
	cudaMalloc(&pointer, bytes);
	#endif
	
	return pointer;
}

static uint64_t cudaGetParameterBufferWrapper(uint64_t alignment, uint64_t bytes)
{
	void* pointer = 0;

	#if EXCLUDE_CUDA_RUNTIME == 0
	cudaMalloc(&pointer, bytes);
	#endif

	uint64_t address = hydrazine::bit_cast<uint64_t>(pointer);
	assert(address % alignment == 0);

	return address;
}

static void cudaFreeWrapper(void* pointer)
{
	#if EXCLUDE_CUDA_RUNTIME == 0
	cudaFree(pointer);
	#endif
}

static size_t align(size_t address, size_t alignment)
{
	size_t remainder = address % alignment;
	return remainder == 0 ? address : address + alignment - remainder;
}

template<typename T>
void parse(std::string& result, size_t& parameters,
	std::ios_base& (*format)(std::ios_base&) = std::dec)
{
	std::stringstream stream;
	parameters = align(parameters, sizeof(T));
	stream << format << *(T*)parameters;
	result.append(stream.str());
	parameters += sizeof(T);
}

static int printfWrapper(size_t string, size_t parameters)
{
	const char* format = (const char*)string;
	std::string result;

	bool escape = false;
	for(const char* f = format; *f != 0; ++f)
	{
		if(escape)
		{
			if(*f == 'c')
			{
				parse<char>(result, parameters);
			}
			else if(*f == 'd' || *f == 'i')
			{
				parse<int>(result, parameters);
			}
			else if (*f == 'f')
			{
				parse<float>(result, parameters);			
			}
			else if (*f == 'o')
			{
				parse<unsigned>(result, parameters, std::oct);			
			}
			else if (*f == 's')
			{
				parameters = align(parameters, sizeof(long long unsigned int));
				std::string temp(*(const char**)parameters);
				result += temp;
				parameters += sizeof(long long unsigned int); // gpus use 64-bit
			}
			else if (*f == 'u')
			{
				parse<unsigned>(result, parameters);			
			}
			else if (*f == 'x')
			{
				parse<unsigned>(result, parameters, std::hex);
			}
			else if (*f == 'p')
			{
				parse<void*>(result, parameters);
				parameters += sizeof(long long unsigned int) - sizeof(void*);
			}
			else
			{
				result.push_back(*f);
			}

			escape = false;
		}
		else
		{
			if(*f == '%')
			{
				escape = true;
			}
			else
			{
				result.push_back(*f);
			}
		}
	}
	
	std::cout << result;
	
	return 0;
}

namespace ocelot
{
	OcelotRuntime::OcelotRuntime() : _initialized( false )
	{
	
	}
	
	void OcelotRuntime::configure( const api::OcelotConfiguration & c )
	{
		if (c.trace.memoryChecker.enabled)
		{
			report( "Creating memory checker" );
			_memoryChecker.setCheckInitialization(  
				c.trace.memoryChecker.checkInitialization );
			ocelot::addTraceGenerator( _memoryChecker, true );
		}
		if (c.trace.raceDetector.enabled)
		{
			report( "Creating memory race detector" );
			_raceDetector.checkAllWrites( 
				!c.trace.raceDetector.ignoreIrrelevantWrites );
			ocelot::addTraceGenerator( _raceDetector, true );
		}
		if (c.trace.debugger.enabled)
		{
				report("Creating interactive PTX debugger");
				_debugger.filter = c.trace.debugger.kernelFilter;
				_debugger.alwaysAttach = c.trace.debugger.alwaysAttach;
				ocelot::addTraceGenerator(_debugger, true);
		}
		if (c.trace.kernelTimer.enabled) {
			report( "Creating kernel timer" );
			_kernelTimer.outputFile = c.trace.kernelTimer.outputFile;
			ocelot::addTraceGenerator(_kernelTimer, true);
		}

		if (c.optimizations.structuralTransform)
		{
			ocelot::addPTXPass(_structuralTransform);
		}
			
		if (c.optimizations.predicateToSelect)
		{
			ocelot::addPTXPass(_predicationToSelect);		
		}
		
		if (c.optimizations.linearScanAllocation)
		{
			ocelot::addPTXPass(_linearScanAllocation);		
		}
			
		if (c.optimizations.mimdThreadScheduling)
		{
			ocelot::addPTXPass(_mimdThreadScheduling);
		}
			
		if (c.optimizations.syncElimination)
		{
			ocelot::addPTXPass(_syncElimination);
		}

		if (c.optimizations.hoistSpecialValues)
		{
			ocelot::addPTXPass(_hoistSpecialValues);
		}

		if (c.optimizations.simplifyCFG)
		{
			ocelot::addPTXPass(_simplifyCFG);
		}
		
		if (c.optimizations.enforceLockStepExecution)
		{
			ocelot::addPTXPass(_enforceLockStepExecution);
		}
		
		if (c.optimizations.inlining)
		{
			ocelot::addPTXPass(_inliner);
		}

		// add built-in functions
		registerExternalFunction("malloc",  (void*)(cudaMallocWrapper));
		registerExternalFunction("free",    (void*)(cudaFreeWrapper));
		registerExternalFunction("vprintf", (void*)(printfWrapper));

		// CNP support
		registerExternalFunction("cudaGetParameterBuffer",
			(void*)(cudaGetParameterBufferWrapper));
	}

}

#endif

