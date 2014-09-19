/*! \file ExecutableKernel.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date December 22, 2009
	\brief implements a kernel that is executable on some device
*/

#ifndef EXECUTABLE_KERNEL_CPP_INCLUDED
#define EXECUTABLE_KERNEL_CPP_INCLUDED

// C includes
#include <memory.h>

// Ocelot includes
#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/trace/interface/TraceGenerator.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/macros.h>

// Debugging messages
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace executive 
{

ExecutableKernel::ExecutableKernel( const ir::IRKernel& k, 
	executive::Device* d ) : ir::IRKernel( k ), device( d ), 
	_constMemorySize( 0 ), _localMemorySize( 0 ), _globalLocalMemorySize( 0 ),
	_maxThreadsPerBlock( 16384 ), _registerCount( 0 ), _sharedMemorySize( 0 ), 
	_externSharedMemorySize( 0 ), _argumentMemorySize( 0 ),
	_parameterMemorySize( 0 ), _cacheConfiguration(CacheConfigurationDefault)
{
	mapArgumentOffsets();
}

ExecutableKernel::ExecutableKernel( executive::Device* d ) :
	device( d ), _constMemorySize( 0 ), _localMemorySize( 0 ),
	_globalLocalMemorySize( 0 ),  _maxThreadsPerBlock( 16384 ),
	_registerCount( 0 ), _sharedMemorySize( 0 ), _externSharedMemorySize( 0 ), 
	_argumentMemorySize( 0 ), _parameterMemorySize( 0 ),
	_cacheConfiguration(CacheConfigurationDefault)
{
	
}

ExecutableKernel::~ExecutableKernel() 
{

}

bool ExecutableKernel::executable() const
{
	return true;
}

void ExecutableKernel::traceEvent(const trace::TraceEvent & event) const
{
	for(TraceGeneratorVector::const_iterator generator = _generators.begin(); 
		generator != _generators.end(); ++generator) {
		(*generator)->event(event);
	}
}

void ExecutableKernel::tracePostEvent(const trace::TraceEvent & event) const
{
	for(TraceGeneratorVector::const_iterator generator = _generators.begin(); 
		generator != _generators.end(); ++generator) {
		(*generator)->postEvent(event);
	}
}

ir::ExternalFunctionSet::ExternalFunction*
	ExecutableKernel::findExternalFunction(
	const std::string& name) const {
	if(_externals == 0) return 0;
	
	return _externals->find(name);
}

unsigned int ExecutableKernel::constMemorySize() const
{
	return _constMemorySize; 
}

unsigned int ExecutableKernel::localMemorySize() const
{ 
	return _localMemorySize; 
}

unsigned int ExecutableKernel::globalLocalMemorySize() const
{ 
	return _globalLocalMemorySize; 
}

unsigned int ExecutableKernel::maxThreadsPerBlock() const
{
	return _maxThreadsPerBlock; 
}

unsigned int ExecutableKernel::registerCount() const
{ 
	return _registerCount; 
}

unsigned int ExecutableKernel::sharedMemorySize() const 
{ 
	return _sharedMemorySize; 
}


/*! \brief sets the cache configuration of the kernele */
void ExecutableKernel::setCacheConfiguration(CacheConfiguration config) {
	_cacheConfiguration = config;
}

/*! \brief sets the cache configuration of the kernele */
ExecutableKernel::CacheConfiguration ExecutableKernel::getCacheConfiguration()
	const {
	return _cacheConfiguration;
}

unsigned int ExecutableKernel::externSharedMemorySize() const 
{ 
	return _externSharedMemorySize; 
}

unsigned int ExecutableKernel::totalSharedMemorySize() const
{
	return externSharedMemorySize() + sharedMemorySize();
}

unsigned int ExecutableKernel::argumentMemorySize() const 
{ 
	return _argumentMemorySize; 
}

unsigned int ExecutableKernel::parameterMemorySize() const
{
	return _parameterMemorySize;
}

const ir::Dim3& ExecutableKernel::blockDim() const
{
	return _blockDim;
}

const ir::Dim3& ExecutableKernel::gridDim() const
{
	return _gridDim;
}


void ExecutableKernel::setTraceGenerators(const TraceGeneratorVector &traceGenerators) {
	_generators = traceGenerators;
}

void ExecutableKernel::addTraceGenerator(
	trace::TraceGenerator *generator) {
	_generators.push_back(generator);
}

void ExecutableKernel::removeTraceGenerator(
	trace::TraceGenerator *generator) {
	TraceGeneratorVector temp = std::move(_generators);
	for (TraceGeneratorVector::iterator gi = temp.begin(); 
		gi != temp.end(); ++gi) {
		if (*gi != generator) {
			_generators.push_back(*gi);
		}
	}
}

void ExecutableKernel::initializeTraceGenerators() {
	report("ExecutableKernel::initializeTraceGenerators() - " << _generators.size() << " active");
	// notify trace generator(s)
	for (TraceGeneratorVector::iterator it = _generators.begin(); 
		it != _generators.end(); ++it) {
		(*it)->initialize(*this);
	}
}

void ExecutableKernel::finalizeTraceGenerators() {
	report("ExecutableKernel::finalizeTraceGenerators()");

	// notify trace generator(s)
	for (TraceGeneratorVector::iterator it = _generators.begin(); 
		it != _generators.end(); ++it) {
		(*it)->finish();
	}
}


/*!
	\brief compute parameter offsets for parameter data
*/
size_t ExecutableKernel::mapArgumentOffsets() {
	unsigned int size = 0;

	for (ParameterVector::iterator it = arguments.begin();
			it != arguments.end(); ++it) {
		unsigned int misAlignment = size % it->getAlignment();
		size += misAlignment == 0 ? 0 : it->getAlignment() - misAlignment;

		it->offset = size;
		size += it->getSize();
	}

	report("ExecutableKernels::mapArgumentOffsets() - '" << name 
		<< "' - size: " << size << " bytes");

	return size;
}

void ExecutableKernel::setArgumentBlock(const unsigned char *parameter, 
	size_t size) {
	mapArgumentOffsets();

	report("ExecutableKernel::setArgumentBlock() - parameterSize = " << size);

	for (ParameterVector::iterator it = arguments.begin();
		it != arguments.end(); ++it) {
		const unsigned char *ptr = parameter + it->offset;

		for (ir::Parameter::ValueVector::iterator 
			val_it = it->arrayValues.begin();
			val_it != it->arrayValues.end(); 
			++val_it, ptr += it->getElementSize()) {
			
			assert((size_t)ptr - (size_t)parameter
				+ it->getElementSize() <= (size_t)size);
			memcpy(&val_it->val_u64, ptr, it->getElementSize());
		}

		report("Configuring parameter " << it->name 
			<< " - offset: "  << it->offset
			<< " - type: " << it->arrayValues.size() << " x " 
			<< ir::PTXOperand::toString(it->type)
			<< " - value: " << ir::Parameter::value(*it));
	}
}

size_t ExecutableKernel::getArgumentBlock(unsigned char* block, 
	size_t maxSize) const {
	size_t offset = 0;
	for (ParameterVector::const_iterator it = arguments.begin();
		it != arguments.end(); ++it) {
		const ir::Parameter& parameter = *it;
		report("Getting parameter " << parameter.name 
			<< " " 
			<< " - type: " << parameter.arrayValues.size() << " x " 
			<< ir::PTXOperand::toString(parameter.type)
			<< " - value: " << ir::Parameter::value(parameter));

		unsigned char *ptr = block + parameter.offset;
		for (ir::Parameter::ValueVector::const_iterator 
			val_it = parameter.arrayValues.begin();
			val_it != parameter.arrayValues.end(); 
			++val_it, ptr += parameter.getElementSize()) {
			
			memcpy(ptr, &val_it->val_u64, parameter.getElementSize());
		}
		offset = parameter.offset + parameter.getElementSize();
	}
		
	return offset;
}

}

#endif

