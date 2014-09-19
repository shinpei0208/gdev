/*!
	\file KernelTestHarness.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date 14 February 2011
	\brief loads serialized device state and module. Configures device with 'before' state, executes
		a kernel, then compares resulting state to loaded 'after' state.
		
		Reports kernel runtime, correctness of results. Useful for analysis and unit testing
*/

#ifndef OCELOT_UTIL_KERNELTESTHARNESS_H_INCLUDED
#define OCELOT_UTIL_KERNELTESTHARNESS_H_INCLUDED

// C++ includes
#include <fstream>
#include <map>

// Ocelot includes
#include <ocelot/util/interface/ExtractedDeviceState.h>

namespace util {

	class KernelTestHarness {
	public:
		typedef std::map<const void*, const void*> PointerMap;
	public:
	
		KernelTestHarness(std::istream &input);
		~KernelTestHarness();
		
		void execute();
		bool compare(std::ostream& stream);
		void reset();
		
	private:
	
		void _setupExecution();
		void _setupTextures(const ExtractedDeviceState::Module &module);
		void _setupMemory();
		void _setupModule();
	
	private:
		//! extracted device state de-serialized from file
		ExtractedDeviceState state;
		
		//! map of pointers 
		PointerMap pointers;
	
	};

}

#endif

