/*! 
	\file KernelTimer.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date June 29, 2012
	\brief measures the total kernel runtime of an application
*/

// Ocelot includes
#include <ocelot/trace/interface/KernelTimer.h>

#include <ocelot/executive/interface/Device.h>
#include <ocelot/executive/interface/ExecutableKernel.h>

#include <ocelot/api/interface/OcelotConfiguration.h>

// Boost includes
#include <boost/lexical_cast.hpp>

// C++ includes
#include <fstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////////

trace::KernelTimer::KernelTimer(): outputFile("traceKernelTimer.json"), kernel(nullptr), kernelSequenceNumber(0), dynamicInstructions(0) {
	outputFile = api::OcelotConfiguration::get().trace.kernelTimer.outputFile;
}

trace::KernelTimer::~KernelTimer() {
}

void trace::KernelTimer::initialize(const executive::ExecutableKernel& kernel) {
	this->kernel = &kernel;
	dynamicInstructions = 0;
	timer.start();
}

void trace::KernelTimer::event(const TraceEvent &) {
	++dynamicInstructions;
}

void trace::KernelTimer::finish() {
	timer.stop();
	
	double seconds = timer.seconds();
	
	std::ofstream file(outputFile.c_str(), std::ios_base::app);

	const char *appname = std::getenv("APPNAME");
	if (!appname) { appname = kernel->module->path().c_str(); }

	file << "{ \"application\": \"" << appname << "\", ";
	
	const char *trial = std::getenv("TRIALNAME");
	if (trial) {
		file << " \"trial\": \"" << trial << "\", ";
	}
	const char *execution = std::getenv("EXECUTION");
	if (execution) {
		file << " \"execution\": " << boost::lexical_cast<int, const char *>(execution) << ", ";
	}
	
	file
		<< "\"ISA\": \"" << ir::Instruction::toString(kernel->device->properties().ISA) << "\", "
		<< "\"device\": \"" << kernel->device->properties().name << "\", "
		<< "\"kernel\": \"" << kernel->name << "\", "
		<< "\"sequenceNumberInApplication\": " << kernelSequenceNumber++ << ", "
		<< "\"instructions\": " << dynamicInstructions << ", "
		<< "\"kernelRuntime\": " << seconds << " }, " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

