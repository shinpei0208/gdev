/*! \file LLVMRuntimeLibrary.cpp
	\date Wednesday June 16, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief A source file for the library of emulation function available to 
		translated kernels
*/


#ifndef LLVM_RUNTIME_LIBRARY_CPP_INCLUDED
#define LLVM_RUNTIME_LIBRARY_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/Device.h>
#include <ocelot/executive/interface/LLVMExecutableKernel.h>
#include <ocelot/executive/interface/LLVMContext.h>
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/TextureOperations.h>

// Hydrazine Includes
#include <hydrazine/interface/math.h>

// Boost Includes
#include <boost/thread/mutex.hpp>

// Preprocessor Macros

// Report Messages
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

// Print out information when executing atomic operations 
#define REPORT_ATOMIC_OPERATIONS 0

// Only print out register updates from the Nth thread
#define DEBUG_NTH_THREAD_ONLY 1

// Print out PTX instructions as they are executed
#define DEBUG_PTX_INSTRUCTION_TRACE 1

// Print out PTX basic blocks as they are entered
#define DEBUG_PTX_INSTRUCTION_TRACE 1

// The id of the thread to print operations for
#define NTH_THREAD 0

// These symbols need to be exported for LLVM to find them on windows
#ifdef _WIN32
#define OCELOT_INTRINSIC __declspec(dllexport)
#else
#define OCELOT_INTRINSIC
#endif

typedef executive::LLVMModuleManager::KernelAndTranslation::MetaData MetaData;

template < typename T >
static void __report( executive::LLVMContext* context, 
	T value, const bool read )
{
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	unsigned int threadId = context->tid.x + 
		context->tid.y * context->ntid.y +
		context->tid.z * context->ntid.y * context->ntid.x;
	if( threadId == NTH_THREAD )
	{
	#endif		
		std::cout << "Thread (" << context->tid.x << ", " << context->tid.y 
			<< ", " << context->tid.z << ") :   ";
		if( read )
		{
			std::cout << "read ";
		}
		else
		{
			std::cout << "write ";
		}
		std::cout << value << std::endl;
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	}
	#endif
}

static std::string location(const ir::Module* module, unsigned int statement)
{
	if(statement == 0) return "unknown";
	
	ir::Module::StatementVector::const_iterator s_it 
		= module->statements().begin();
	std::advance(s_it, statement);
	ir::Module::StatementVector::const_reverse_iterator s_rit 
		= ir::Module::StatementVector::const_reverse_iterator( s_it );
	unsigned int program = 0;
	unsigned int line = 0;
	unsigned int col = 0;
	for( ; s_rit != module->statements().rend(); ++s_rit ) 
	{
		if( s_rit->directive == ir::PTXStatement::Loc) 
		{
			line = s_rit->sourceLine;
			col = s_rit->sourceColumn;
			program = s_rit->sourceFile;
			break;
		}
	}

	std::string fileName;
	for( s_it = module->statements().begin(); 
		s_it != module->statements().end(); ++s_it ) 
	{
		if(s_it->directive == ir::PTXStatement::File) 
		{
			if(s_it->sourceFile == program) 
			{
				fileName = s_it->name;
				break;
			}
		}
	}

	std::stringstream stream;
	stream << fileName << ":" << line << ":" << col;
	return stream.str();
}

static std::string instruction(const ir::Module* module, unsigned int statement)
{
	if(statement == 0) return "unknown";
	
	assert( statement < module->statements().size() );
	ir::Module::StatementVector::const_iterator s_it 
		= module->statements().begin();
	std::advance( s_it, statement );
	assertM( s_it->instruction.valid() == "", s_it->instruction.valid() );
	return s_it->instruction.toString();
}

static boost::mutex mutex;

extern "C"
{	
	
unsigned int OCELOT_INTRINSIC __ocelot_bfi_b32(
	unsigned int in, unsigned int orig, 
	unsigned int position, unsigned int length )
{
	return hydrazine::bitFieldInsert( in, orig, position, length );
}

long long unsigned int OCELOT_INTRINSIC __ocelot_bfi_b64(
	long long unsigned int in, 
	long long unsigned int orig, unsigned int position, 
	unsigned int length )
{
	return hydrazine::bitFieldInsert( in, orig, position, length );
}

unsigned int OCELOT_INTRINSIC __ocelot_bfe_b32(
	unsigned int a, unsigned int pos,
    unsigned int len, bool isSigned )
{
	return hydrazine::bfe( a, pos, len, isSigned );
}

long long unsigned int OCELOT_INTRINSIC __ocelot_bfe_b64(
	long long unsigned int a, 
    unsigned int pos, unsigned int len, bool isSigned )
{
	return hydrazine::bfe( a, pos, len, isSigned );
}

unsigned int OCELOT_INTRINSIC __ocelot_bfind_b32( unsigned int a, bool shift )
{
	return hydrazine::bfind( a, shift );
}

long long unsigned int OCELOT_INTRINSIC __ocelot_bfind_b64(
	long long unsigned int a, bool shift )
{
	return hydrazine::bfind( a, shift );
}

unsigned int OCELOT_INTRINSIC __ocelot_brev_b32( unsigned int a )
{
	return hydrazine::brev( a );
}

long long unsigned int OCELOT_INTRINSIC __ocelot_brev_b64(
	long long unsigned int a )
{
	return hydrazine::brev( a );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt( unsigned int a,
	unsigned int b, unsigned int c )
{
	return hydrazine::permute< hydrazine::DefaultPermute >( a, b, c );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt_f4e( unsigned int a, unsigned int b, 
	unsigned int c )
{
	return hydrazine::permute< hydrazine::ForwardFourExtract >( a, b, c );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt_b4e( unsigned int a, unsigned int b, 
	unsigned int c )
{
	return hydrazine::permute< hydrazine::BackwardFourExtract >( a, b, c );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt_rc8( unsigned int a, unsigned int b, 
	unsigned int c )
{
	return hydrazine::permute< hydrazine::ReplicateEight >( a, b, c );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt_ecl( unsigned int a, unsigned int b, 
	unsigned int c )
{
	return hydrazine::permute< hydrazine::EdgeClampLeft >( a, b, c );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt_ecr(
	unsigned int a, unsigned int b, unsigned int c )
{
	return hydrazine::permute< hydrazine::EdgeClampRight >( a, b, c );
}

unsigned int OCELOT_INTRINSIC __ocelot_prmt_rc16(
	unsigned int a, unsigned int b, unsigned int c )
{
	return hydrazine::permute< hydrazine::ReplicateSixteen >( a, b, c );
}

long long int OCELOT_INTRINSIC __ocelot_mul_hi_s64(
	long long int a, long long int b )
{
	long long int hi = 0;
	long long int lo = 0;
	
	hydrazine::multiplyHiLo( hi, lo, a, b );
	
	return hi;
}

long long unsigned int OCELOT_INTRINSIC __ocelot_mul_hi_u64(
	long long unsigned int a, 
	long long unsigned int b )
{
	long long unsigned int hi = 0;
	long long unsigned int lo = 0;
	
	hydrazine::multiplyHiLo( hi, lo, a, b );
	
	return hi;
}

bool OCELOT_INTRINSIC __ocelot_vote( bool a,
	ir::PTXInstruction::VoteMode mode, bool invert )
{
	a = invert ? !a : a;
	
	switch( mode )
	{
		case ir::PTXInstruction::All:
		case ir::PTXInstruction::Any:
		{
			return a;
			break;
		}
		case ir::PTXInstruction::Uni:
		default: break;
	}
	return true;
}

ir::PTXB32 OCELOT_INTRINSIC __ocelot_atomic_inc_32(
	ir::PTXU64 address, ir::PTXB32 b )
{
	ir::PTXB32 d = 0;
	ir::PTXB32 result = 0;

	mutex.lock();

	d = *((ir::PTXB32*) address);
	
	result = (d >= b) ? 0 : d + 1;
	reportE( REPORT_ATOMIC_OPERATIONS, "AtomicInc: address " 
		<< (void*) address << " from " << d << " by " << b 
		<< " to " << result );

	*((ir::PTXB32*) address) = result;
	
	mutex.unlock();

	return d;
}

ir::PTXB32 OCELOT_INTRINSIC __ocelot_atomic_dec_32(
	ir::PTXU64 address, ir::PTXB32 b )
{
	ir::PTXB32 d = 0;
	ir::PTXB32 result = 0;

	mutex.lock();

	d = *((ir::PTXB32*) address);
	
	result = ((d == 0) || (d > b)) ? b : d - 1;
	reportE( REPORT_ATOMIC_OPERATIONS, "AtomicDec: address " 
		<< (void*) address << " from " << d << " by " << b 
		<< " to " << result );

	*((ir::PTXB32*) address) = result;
	
	mutex.unlock();

	return d;
}

void OCELOT_INTRINSIC __ocelot_debug_block( executive::LLVMContext* context, 
	ir::ControlFlowGraph::BasicBlock::Id id )
{
	#if(DEBUG_PTX_BASIC_BLOCK_TRACE == 1)
	MetaData* state = (MetaData*) context->metadata;
	
	MetaData::BlockIdMap::const_iterator
		block = state->blocks.find( id );
	assert( block != state->blocks.end() );
	
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	unsigned int threadId = context->tid.x + 
		context->tid.y * context->ntid.y +
		context->tid.z * context->ntid.y * context->ntid.x;
	if( threadId == NTH_THREAD )
	{
	#endif
	
	std::cout << "Thread (" << context->tid.x << ", " << context->tid.y 
		<< ", " << context->tid.z << ") : Basic Block \"" << std::flush;
	std::cout << block->second->label << "\"\n";

	#if(DEBUG_NTH_THREAD_ONLY == 1)
	}
	#endif
	#endif
}

void OCELOT_INTRINSIC __ocelot_debug_instruction( executive::LLVMContext* context, 
	ir::PTXU64 _instruction )
{
	#if(DEBUG_PTX_INSTRUCTION_TRACE == 1)		
	void* instruction = (void*) _instruction;

	#if(DEBUG_NTH_THREAD_ONLY == 1)
	unsigned int threadId = context->tid.x + 
		context->tid.y * context->ntid.y +
		context->tid.z * context->ntid.y * context->ntid.x;
	if( threadId == NTH_THREAD )
	{
	#endif
	
	std::cout << "Thread (" << context->tid.x << ", " << context->tid.y 
		<< ", " << context->tid.z << ") :  " << std::flush;
	std::cout << static_cast<ir::Instruction*>(instruction)->toString() 
		<< "\n";

	#if(DEBUG_NTH_THREAD_ONLY == 1)
	}
	#endif

	#endif
}

void OCELOT_INTRINSIC __ocelot_register_write_s8(
	executive::LLVMContext* context, ir::PTXS8 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_s16(
	executive::LLVMContext* context, ir::PTXS16 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_s32(
	executive::LLVMContext* context, ir::PTXS32 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_s64(
	executive::LLVMContext* context, ir::PTXS64 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_u8(
	executive::LLVMContext* context, ir::PTXU8 value )
{
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	unsigned int threadId = context->tid.x + 
		context->tid.y * context->ntid.y +
		context->tid.z * context->ntid.y * context->ntid.x;
	if( threadId == NTH_THREAD )
	{
	#endif		
		std::cout << "Thread (" << context->tid.x << ", " << context->tid.y 
			<< ", " << context->tid.z << ") :  ";
		std::cout << " write ";
		std::cout << (unsigned int)value << std::endl;
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	}
	#endif
}

void OCELOT_INTRINSIC __ocelot_register_write_u16(
	executive::LLVMContext* context, 
	ir::PTXU16 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_u32(
	executive::LLVMContext* context, 
	ir::PTXU32 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_u64(
	executive::LLVMContext* context, 
	ir::PTXU64 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_f32(
	executive::LLVMContext* context, 
	ir::PTXF32 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_write_f64(
	executive::LLVMContext* context, 
	ir::PTXF64 value )
{
	__report( context, value, false );
}

void OCELOT_INTRINSIC __ocelot_register_read_s8(
	executive::LLVMContext* context, 
	ir::PTXS8 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_s16(
	executive::LLVMContext* context, 
	ir::PTXS16 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_s32(
	executive::LLVMContext* context, 
	ir::PTXS32 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_s64(
	executive::LLVMContext* context, 
	ir::PTXS64 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_u8(
	executive::LLVMContext* context, 
	ir::PTXU8 value )
{
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	unsigned int threadId = context->tid.x + 
		context->tid.y * context->ntid.y +
		context->tid.z * context->ntid.y * context->ntid.x;
	if( threadId == NTH_THREAD )
	{
	#endif		
		std::cout << "Thread (" << context->tid.x << ", " << context->tid.y 
			<< ", " << context->tid.z << ") :  ";
		std::cout << " read ";
		std::cout << (unsigned int)value << std::endl;
	#if(DEBUG_NTH_THREAD_ONLY == 1)
	}
	#endif
}

void OCELOT_INTRINSIC __ocelot_register_read_u16(
	executive::LLVMContext* context, ir::PTXU16 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_u32(
	executive::LLVMContext* context, 
	ir::PTXU32 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_u64(
	executive::LLVMContext* context, 
	ir::PTXU64 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_f32(
	executive::LLVMContext* context, 
	ir::PTXF32 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_register_read_f64(
	executive::LLVMContext* context, 
	ir::PTXF64 value )
{
	__report( context, value, true );
}

void OCELOT_INTRINSIC __ocelot_check_global_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 _address, unsigned int bytes, unsigned int statement )
{
	void* address = (void*)_address;
	MetaData* state = (MetaData*) context->metadata;
	
	#if 0
	
	if( !state->device->checkMemoryAccess( address, bytes ) )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;
		
		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Global memory address " 
			<< address << " of size " << bytes
			<< " is out of any allocated or mapped range.\n";
		std::cerr << "Memory Map:\n";
		std::cerr << 
			state->device->nearbyAllocationsToString( address );
		std::cerr << "\n";
		std::cout << "\tNear: " << location( state->kernel->module, statement )
			<< "\n\n";
		assertM(false, "Aborting execution.");
	}
	
	#endif
	
	bool error = bytes == 0;
	if( !error ) error = (long long unsigned int)address % bytes != 0;
	
	if( error )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;

		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Global memory address " 
			<< address << " of size " << bytes
			<< " is not aligned to the access size.\n";
		std::cerr << "\n";
		std::cout << "\tNear: "
			<< location( state->kernel->module, statement ) << "\n\n";
		assertM(false, "Aborting execution.");
	}
}

void OCELOT_INTRINSIC __ocelot_check_shared_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 _address, unsigned int bytes, unsigned int statement )
{
	MetaData* state = (MetaData*) context->metadata;
	
	char* address = (char*) _address;
	char* end = address + bytes;
	char* allocationEnd = context->shared + state->sharedSize
		+ context->externalSharedSize;
	
	if( end > allocationEnd )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;
		
		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Shared memory address " 
			<< _address << " is " << (end - allocationEnd)
			<< " bytes beyond the shared memory block of " 
			<< state->sharedSize << " bytes.\n";
		std::cout << "\tNear: "
			<< location( state->kernel->module, statement ) << "\n\n";
		assertM(false, "Aborting execution.");
	}
}

void OCELOT_INTRINSIC __ocelot_check_constant_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 _address, unsigned int bytes, unsigned int statement )
{
	MetaData* state = (MetaData*) context->metadata;
	
	char* address = (char*) _address;
	char* end = address + bytes;
	char* allocationEnd = context->constant + state->constantSize;
	
	if( end > allocationEnd )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;
		
		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Constant memory address " 
			<< _address << " = " << (void *)_address << " of size " 
				<< bytes << " bytes is " << (end - allocationEnd)
			<< " bytes beyond the constant memory block of " 
			<< state->constantSize << " bytes\n  on interval: " 
			<< (void *)context->constant 
			<< " - " << (void *)allocationEnd << "\n";
		std::cout << "\tNear: "
			<< location( state->kernel->module, statement ) << "\n\n";
		assertM(false, "Aborting execution.");
	}	
}

void OCELOT_INTRINSIC __ocelot_check_local_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 _address, unsigned int bytes, unsigned int statement )
{
	MetaData* state = (MetaData*) context->metadata;
	
	char* address = (char*) _address;
	char* end = address + bytes;
	char* allocationEnd  = context->local + state->localSize;
	char* globalLocalEnd = context->globallyScopedLocal +
		state->globalLocalSize;
	
	bool inLocal = (address < allocationEnd) && (address >= context->local);
	bool inGlobalLocal = (address < globalLocalEnd) &&
		(address >= context->globallyScopedLocal);
	
	if( !inLocal && !inGlobalLocal )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;
		
		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Local memory address " 
			<< _address << " is " << (end - allocationEnd)
			<< " bytes beyond the local memory block of " 
			<< state->localSize << " bytes.\n";
		std::cout << "\tNear: "
			<< location( state->kernel->module, statement ) << "\n\n";
		assertM(false, "Aborting execution.");
	}	
}

void OCELOT_INTRINSIC __ocelot_check_param_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 _address, unsigned int bytes, unsigned int statement )
{
	MetaData* state = (MetaData*) context->metadata;
	
	char* address = (char*) _address;
	char* end = address + bytes;
	char* allocationEnd = context->parameter + state->parameterSize;
			
	if( end > allocationEnd )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;
		
		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Parameter memory address " 
			<< address << " is  " << (end - allocationEnd)
			<< " bytes beyond the parameter memory block of " 
			<< state->parameterSize << " bytes.\n";
		std::cout << "\tNear: "
			<< location( state->kernel->module, statement ) << "\n\n";
		assertM(false, "Aborting execution.");
	}	
}

void OCELOT_INTRINSIC __ocelot_check_argument_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 _address, unsigned int bytes, unsigned int statement )
{
	MetaData* state = (MetaData*) context->metadata;
	
	char* address = (char*) _address;
	char* end = address + bytes;
	char* allocationEnd = context->argument + state->argumentSize;
			
	if( end > allocationEnd )
	{
		unsigned int thread = context->tid.x 
			+ context->ntid.x * context->tid.y 
			+ context->ntid.x * context->ntid.y * context->tid.y;
		unsigned int cta = context->ctaid.x 
			+ context->nctaid.x * context->ctaid.y 
			+ context->nctaid.x * context->nctaid.y * context->ctaid.y;
		
		std::cerr << "While executing kernel '" 
			<< state->kernel->name << "'\n";
		std::cerr << "Error in (cta " << cta << ")(thread " << thread 
			<< "): instruction '" 
			<< instruction( state->kernel->module, statement ) << "'\n";
		std::cerr << "Argument memory address " 
			<< address << " is  " << (end - allocationEnd)
			<< " bytes beyond the argument memory block of " 
			<< state->argumentSize << " bytes.\n";
		std::cout << "\tNear: "
			<< location( state->kernel->module, statement ) << "\n\n";
		assertM(false, "Aborting execution.");
	}	
}

void OCELOT_INTRINSIC __ocelot_check_generic_memory_access(
	executive::LLVMContext* context,
	ir::PTXU64 address, unsigned int bytes, unsigned int statement )
{
	// TODO check this correctly
	__ocelot_check_global_memory_access( context, address,
		bytes, statement );
}
	
void OCELOT_INTRINSIC __ocelot_tex_3d_fs( float* result,
	executive::LLVMContext* context, 
	unsigned int index, unsigned int c0, unsigned int c1, unsigned int c2,
	unsigned int c3 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, float >( 
		texture, c0, c1, c2 );
	result[1] = executive::tex::sample< 1, float >( 
		texture, c0, c1, c2 );
	result[2] = executive::tex::sample< 2, float >( 
		texture, c0, c1, c2 );
	result[3] = executive::tex::sample< 3, float >( 
		texture, c0, c1, c2 );
}

void OCELOT_INTRINSIC __ocelot_tex_3d_fu( float* result,
	executive::LLVMContext* context, unsigned int index, unsigned int c0,
	unsigned int c1, unsigned int c2, unsigned int c3 )
{
	__ocelot_tex_3d_fs( result, context, index, c0, c1, c2, c3 );
}

void OCELOT_INTRINSIC __ocelot_tex_3d_ff( float* result,
	executive::LLVMContext* context, 
	unsigned int index, float c0, float c1, float c2, float c3 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, float >( 
		texture, c0, c1, c2 );
	result[1] = executive::tex::sample< 1, float >( 
		texture, c0, c1, c2 );
	result[2] = executive::tex::sample< 2, float >( 
		texture, c0, c1, c2 );
	result[3] = executive::tex::sample< 3, float >( 
		texture, c0, c1, c2 );	
}

void OCELOT_INTRINSIC __ocelot_tex_3d_sf( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, float c0, 
	float c1, float c2, float c3 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, int >( 
		texture, c0, c1, c2 );
	result[1] = executive::tex::sample< 1, int >( 
		texture, c0, c1, c2 );
	result[2] = executive::tex::sample< 2, int >( 
		texture, c0, c1, c2 );
	result[3] = executive::tex::sample< 3, int >( 
		texture, c0, c1, c2 );				
}

void OCELOT_INTRINSIC __ocelot_tex_3d_uf( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, float c0, 
	float c1, float c2, float c3 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, unsigned int >( 
		texture, c0, c1, c2 );
	result[1] = executive::tex::sample< 1, unsigned int >( 
		texture, c0, c1, c2 );
	result[2] = executive::tex::sample< 2, unsigned int >( 
		texture, c0, c1, c2 );
	result[3] = executive::tex::sample< 3, unsigned int >( 
		texture, c0, c1, c2 );				
}

void OCELOT_INTRINSIC __ocelot_tex_3d_su( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0, 
	unsigned int c1, unsigned int c2, unsigned int c3 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, int >( 
		texture, c0, c1, c2 );
	result[1] = executive::tex::sample< 1, int >( 
		texture, c0, c1, c2 );
	result[2] = executive::tex::sample< 2, int >( 
		texture, c0, c1, c2 );
	result[3] = executive::tex::sample< 3, int >( 
		texture, c0, c1, c2 );				
}

void OCELOT_INTRINSIC __ocelot_tex_3d_ss( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0, 
	unsigned int c1, unsigned int c2, unsigned int c3 )
{
	__ocelot_tex_3d_su( result, context, index, c0, c1, c2, c3 );
}

void OCELOT_INTRINSIC __ocelot_tex_3d_uu( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0, 
	unsigned int c1, unsigned int c2, unsigned int c3 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, unsigned int >( 
		texture, c0, c1, c2 );
	result[1] = executive::tex::sample< 1, unsigned int >( 
		texture, c0, c1, c2 );
	result[2] = executive::tex::sample< 2, unsigned int >( 
		texture, c0, c1, c2 );
	result[3] = executive::tex::sample< 3, unsigned int >( 
		texture, c0, c1, c2 );				
}

void OCELOT_INTRINSIC __ocelot_tex_3d_us( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0, 
	unsigned int c1, unsigned int c2, unsigned int c3 )
{
	__ocelot_tex_3d_uu( result, context, index, c0, c1, c2, c3 );
}

void OCELOT_INTRINSIC __ocelot_tex_2d_fu( float* result,
	executive::LLVMContext* context, 
	unsigned int index, unsigned int c0, unsigned int c1 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, float >( texture, c0, c1 );
	result[1] = executive::tex::sample< 1, float >( texture, c0, c1 );
	result[2] = executive::tex::sample< 2, float >( texture, c0, c1 );
	result[3] = executive::tex::sample< 3, float >( texture, c0, c1 );		
}

void OCELOT_INTRINSIC __ocelot_tex_2d_fs( float* result,
	executive::LLVMContext* context, 
	unsigned int index, unsigned int c0, unsigned int c1 )
{
	__ocelot_tex_2d_fu( result, context, index, c0, c1 );
}

void OCELOT_INTRINSIC __ocelot_tex_2d_ff( float* result,
	executive::LLVMContext* context, 
	unsigned int index, float c0, float c1 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, float >( texture, c0, c1 );
	result[1] = executive::tex::sample< 1, float >( texture, c0, c1 );
	result[2] = executive::tex::sample< 2, float >( texture, c0, c1 );
	result[3] = executive::tex::sample< 3, float >( texture, c0, c1 );	
}

void OCELOT_INTRINSIC __ocelot_tex_a2d_ff( float* result,
    executive::LLVMContext* context, 
    unsigned int index, float c0, float c1, int offset)
{
    //index += offset * 512;
    __ocelot_tex_2d_ff( result, context, index, c0, c1 );
}

void OCELOT_INTRINSIC __ocelot_tex_2d_sf( unsigned int* result, 
	executive::LLVMContext* context, 
	unsigned int index, float c0, float c1 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, int >( texture, c0, c1 );
	result[1] = executive::tex::sample< 1, int >( texture, c0, c1 );
	result[2] = executive::tex::sample< 2, int >( texture, c0, c1 );
	result[3] = executive::tex::sample< 3, int >( texture, c0, c1 );
}

void OCELOT_INTRINSIC __ocelot_tex_2d_uf( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, 
	float c0, float c1 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, 
		unsigned int >( texture, c0, c1 );
	result[1] = executive::tex::sample< 1, 
		unsigned int >( texture, c0, c1 );
	result[2] = executive::tex::sample< 2, 
		unsigned int >( texture, c0, c1 );
	result[3] = executive::tex::sample< 3, 
		unsigned int >( texture, c0, c1 );						
}

void OCELOT_INTRINSIC __ocelot_tex_2d_us( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0, 
	unsigned int c1 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, 
		unsigned int >( texture, c0, c1 );
	result[1] = executive::tex::sample< 1, 
		unsigned int >( texture, c0, c1 );
	result[2] = executive::tex::sample< 2, 
		unsigned int >( texture, c0, c1 );
	result[3] = executive::tex::sample< 3, 
		unsigned int >( texture, c0, c1 );						
}

void OCELOT_INTRINSIC __ocelot_tex_2d_uu( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, 
	unsigned int c0, unsigned int c1 )
{
	__ocelot_tex_2d_us( result, context, index, c0, c1 );
}

void OCELOT_INTRINSIC __ocelot_tex_2d_su( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, 
	unsigned int c0, unsigned int c1 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, int >( texture, c0, c1 );
	result[1] = executive::tex::sample< 1, int >( texture, c0, c1 );
	result[2] = executive::tex::sample< 2, int >( texture, c0, c1 );
	result[3] = executive::tex::sample< 3, int >( texture, c0, c1 );	
}

void OCELOT_INTRINSIC __ocelot_tex_2d_ss( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, 
	unsigned int c0, unsigned int c1 )
{
	__ocelot_tex_2d_su( result, context, index, c0, c1 );
}

void OCELOT_INTRINSIC __ocelot_tex_1d_fs( float* result,
	executive::LLVMContext* context, 
	int index, int c0 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, float >( texture, c0 );
	result[1] = executive::tex::sample< 1, float >( texture, c0 );
	result[2] = executive::tex::sample< 2, float >( texture, c0 );
	result[3] = executive::tex::sample< 3, float >( texture, c0 );	
}

void OCELOT_INTRINSIC __ocelot_tex_1d_fu( float* result,
	executive::LLVMContext* context, 
	unsigned int index, unsigned int c0 )
{
	__ocelot_tex_1d_fs( result, context, index, c0 );
}

void OCELOT_INTRINSIC __ocelot_tex_1d_ff( float* result,
	executive::LLVMContext* context, 
	unsigned int index, float c0 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, float >( texture, c0 );
	result[1] = executive::tex::sample< 1, float >( texture, c0 );
	result[2] = executive::tex::sample< 2, float >( texture, c0 );
	result[3] = executive::tex::sample< 3, float >( texture, c0 );		
}

void OCELOT_INTRINSIC __ocelot_tex_1d_sf( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, float c0 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, int >( texture, c0 );
	result[1] = executive::tex::sample< 1, int >( texture, c0 );
	result[2] = executive::tex::sample< 2, int >( texture, c0 );
	result[3] = executive::tex::sample< 3, int >( texture, c0 );	
}

void OCELOT_INTRINSIC __ocelot_tex_1d_uf( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, float c0 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, unsigned int >( texture, c0 );
	result[1] = executive::tex::sample< 1, unsigned int >( texture, c0 );
	result[2] = executive::tex::sample< 2, unsigned int >( texture, c0 );
	result[3] = executive::tex::sample< 3, unsigned int >( texture, c0 );	
}

void OCELOT_INTRINSIC __ocelot_tex_1d_ss( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0 )
{
	MetaData* state = (MetaData*) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, int >( texture, c0 );
	result[1] = executive::tex::sample< 1, int >( texture, c0 );
	result[2] = executive::tex::sample< 2, int >( texture, c0 );
	result[3] = executive::tex::sample< 3, int >( texture, c0 );	
}

void OCELOT_INTRINSIC __ocelot_tex_1d_su( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0 )
{
	__ocelot_tex_1d_ss( result, context, index, c0 );
}

void OCELOT_INTRINSIC __ocelot_tex_1d_us( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0 )
{
	MetaData* state = ( MetaData* ) context->metadata;
	const ir::Texture& texture = *state->textures[ index ];
	
	result[0] = executive::tex::sample< 0, unsigned int >( texture, c0 );
	result[1] = executive::tex::sample< 1, unsigned int >( texture, c0 );
	result[2] = executive::tex::sample< 2, unsigned int >( texture, c0 );
	result[3] = executive::tex::sample< 3, unsigned int >( texture, c0 );
	
}

void OCELOT_INTRINSIC __ocelot_tex_1d_uu( unsigned int* result, 
	executive::LLVMContext* context, unsigned int index, unsigned int c0 )
{
	__ocelot_tex_1d_us( result, context, index, c0 );
}

unsigned int OCELOT_INTRINSIC __ocelot_get_extent(
	executive::LLVMContext* context,
	unsigned int space )
{
	MetaData* state = ( MetaData* ) context->metadata;
	
	switch( ( ir::PTXInstruction::AddressSpace ) space )
	{
		case ir::PTXInstruction::Local:
		{
			report("Local memory size is " << state->localSize);
			return state->localSize;
		}
		case ir::PTXInstruction::Param:
		{
			report("Parameter memory size is " << state->parameterSize);
			return state->parameterSize;
		}
		case ir::PTXInstruction::Shared:
		{
			report("Shared memory size is " << state->sharedSize);
			return state->sharedSize;
		}
		default: assertM( false, "Invalid memory space." );
	}
	
	return 0;
}

}

#endif

