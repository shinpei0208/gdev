/*! \file TraceGenerator.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief implements the base TraceGenerator class
*/

#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/executive/interface/EmulatedKernel.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

trace::TraceGenerator::TraceGenerator() {

}

trace::TraceGenerator::~TraceGenerator() {

}

void trace::TraceGenerator::initialize(
	const executive::ExecutableKernel& kernel) {
	// if we cared, we could get the kernel's launch configuration:
	//
	// kernel->gridDim
	// kernel->blockDim
	// kernel->threadCount
	// kernel->RegisterCount
	// kernel->ParameterMemorySize
	// kernel->ConstMemorySize
	// kernel->SharedMemorySize
	// kernel->KernelInstructions
	report( "Initializing trace generator for kernel " << kernel.name );
}

void trace::TraceGenerator::event(const TraceEvent & event) {
	// do something meaningful with the trace
	report( "Default Event(): " << event.toString() );
}

void trace::TraceGenerator::postEvent(const TraceEvent & event) {
	// do something meaningful with the trace
	report( "Default PostEvent(): " << event.toString() );
}

void trace::TraceGenerator::finish() {
}

