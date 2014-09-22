/*! \file MemoryChecker.cpp
	\date Wednesday March 17, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the MemoryChecker class.
*/

#ifndef MEMORY_CHECKER_CPP_INCLUDED
#define MEMORY_CHECKER_CPP_INCLUDED

// Ocelot includes
#include <ocelot/trace/interface/MemoryChecker.h>
#include <ocelot/trace/interface/TraceEvent.h>
#include <ocelot/executive/interface/EmulatedKernel.h>
#include <ocelot/executive/interface/Device.h>

// hydrazine includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// Debugging messages
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0
#define REPORT_KERNEL_INSTRUCTIONS 0

namespace trace
{
	MemoryChecker::Allocation::Allocation( bool v, ir::PTXU64 b, ir::PTXU64 e )
		: valid( v ), base( b ), extent( e )
	{
	
	}
	
	static std::string prefix( unsigned int thread, const ir::Dim3& dim, 
		const TraceEvent& e )
	{
		const unsigned int cta = e.blockId.x + e.blockId.y * dim.x 
			+ dim.x * dim.y * e.blockId.z;
		
		std::stringstream stream;
		stream << "[PC " << e.PC << "] ";
		stream << "[thread " << thread << "] ";
		stream << "[cta " << cta << "] ";
		stream << e.instruction->toString() << " - ";
		
		return stream.str();
	}
	
	static void alignmentError( const executive::EmulatedKernel* kernel, 
		const ir::Dim3& dim, const TraceEvent& e, unsigned int thread, 
		ir::PTXU64 address, unsigned int size )
	{
		std::stringstream stream;
		stream << prefix( thread, dim, e );
		stream << "Memory access " << (void*)address 
			<< " is not aligned to the access size ( " << size << " bytes )";
		stream << "\n";
		stream << "Near " << kernel->location( e.PC ) << "\n";
		throw hydrazine::Exception( stream.str() );
	}
	
	template< unsigned int size >
	static void checkAlignment( const executive::EmulatedKernel* kernel, 
		const ir::Dim3& dim, const TraceEvent& e )
	{
		TraceEvent::U64Vector::const_iterator 
			address = e.memory_addresses.begin();
		unsigned int threads = e.active.size();
		for( unsigned int thread = 0; thread < threads; ++thread )
		{
			if( !e.active[ thread ] ) continue;
			if( *address & ( size - 1 ) )
			{
				alignmentError( kernel, dim, e, thread, *address, size );
			}
			++address;
		}
	}
	
	void MemoryChecker::_checkAlignment( const TraceEvent& e )
	{		
		switch( e.memory_size )
		{
			case 1: return;
			case 2: checkAlignment< 2 >( _kernel, _dim, e ); break;
			case 4: checkAlignment< 4 >( _kernel, _dim, e ); break;
			case 8: checkAlignment< 8 >( _kernel, _dim, e ); break;
			case 16: checkAlignment< 16 >( _kernel, _dim, e ); break;
			case 32: checkAlignment< 32 >( _kernel, _dim, e ); break;
			default: break;
		}
	}

	static void uninitError( const ir::Dim3& dim,
		unsigned int thread, const TraceEvent& event, 
		const executive::EmulatedKernel* kernel )
	{
		std::stringstream stream;
		stream << prefix( thread, dim, event );
		stream << "Control flow directed by undefined value";
		stream << "\n";
		stream << "Near " << kernel->location( event.PC ) << "\n";
		throw hydrazine::Exception( stream.str() );
	}
	
	static void memoryUninitError( const std::string& space, const ir::Dim3& dim, 
		const executive::Device* device, 
		unsigned int thread, ir::PTXU64 address, 
		unsigned int size, const TraceEvent& event, 
		const executive::EmulatedKernel* kernel )
	{
		std::stringstream stream;
		stream << prefix( thread, dim, event );
		stream << space << " memory access " << (void*)address 
			<< " to an uninitialized memory location.\n\n";
		stream << "Nearby Device Allocations\n";
		stream << device->nearbyAllocationsToString( (void*)address );
		stream << "\n";
		stream << "Near " << kernel->location( event.PC ) << "\n";
		throw hydrazine::Exception( stream.str() );
	}
	
	static void memoryError( const std::string& space, const ir::Dim3& dim, 
		unsigned int thread, ir::PTXU64 address, unsigned int size, 
		const TraceEvent& e, const executive::EmulatedKernel* kernel, 
		unsigned int extent )
	{
		std::stringstream stream;
		stream << prefix( thread, dim, e );
		stream << space << " memory access " << (void*) address 
			<< " (" << size << " bytes) is beyond allocated block size " 
			<< extent;
		stream << "\n";
		stream << "Near " << kernel->location( e.PC ) << "\n";
		throw hydrazine::Exception( stream.str() );
	}

	static void globalMemoryError( const executive::Device* device, 
		const ir::Dim3& dim, unsigned int thread, ir::PTXU64 address, 
		unsigned int size, const TraceEvent& event, 
		const executive::EmulatedKernel* kernel )
	{
		std::stringstream stream;
		stream << prefix( thread, dim, event );
		stream << "Global memory access " << (void*)address 
			<< " is not within any allocated or mapped range.\n\n";
		stream << "Nearby Device Allocations\n";
		stream << device->nearbyAllocationsToString( (void*)address );
		stream << "\n";
		stream << "Near " << kernel->location( event.PC ) << "\n";
		throw hydrazine::Exception( stream.str() );
	}

	static void checkLocalAccess( const std::string& space, const ir::Dim3& dim,
		ir::PTXU64 base, ir::PTXU64 extent,
		const TraceEvent& event, const executive::EmulatedKernel* kernel )
	{
		TraceEvent::U64Vector::const_iterator 
			address = event.memory_addresses.begin();

		unsigned int threads = event.active.size();
		for( unsigned int thread = 0; thread < threads; ++thread )
		{
			if( !event.active[ thread ] ) continue;
			if( base > *address || *address >= base + extent )
			{
				memoryError( space, dim, thread, 
					*address, event.memory_size, event, kernel, extent );
			}
			++address;
		}
	}

	void MemoryChecker::_checkValidity( const TraceEvent& e )
	{
		switch( e.instruction->addressSpace )
		{
			case ir::PTXInstruction::Generic:
			{
				TraceEvent::U64Vector::const_iterator 
					address = e.memory_addresses.begin();

				unsigned int threads = e.active.size();
				for( unsigned int thread = 0; thread < threads; ++thread )
				{
					if( !e.active[ thread ] ) continue;
					if(_kernel->totalSharedMemorySize() > 0)
					{
						if( (ir::PTXU64)_kernel->getSharedMemory() <= *address 
							&& *address < (ir::PTXU64)_kernel->getSharedMemory()
							+ _shared.extent )
						{
							++address;
							continue;
						}
					}
					if( (ir::PTXU64)_kernel->getStackBase(thread) <= *address 
						&& *address < (ir::PTXU64)_kernel->getStackBase(thread)
						+ (ir::PTXU64)_kernel->getTotalStackSize(thread))
					{
						++address;
						continue;
					}
					if( (ir::PTXU64)_kernel->getGlobalLocalMemory(thread)
						<= *address && *address
						< (ir::PTXU64)_kernel->getGlobalLocalMemory(thread)
						+ _globalLocal.extent )
					{
						++address;
						continue;
					}
					if( (ir::PTXU64)_kernel->ConstMemory <= *address 
						&& *address < (ir::PTXU64)_kernel->ConstMemory +
						_constant.extent)
					{
						++address;
						continue;
					}
					if( _cache.base > *address 
						|| *address >= _cache.base + _cache.extent
						|| !_cache.valid )
					{
						const executive::Device::MemoryAllocation* allocation = 
							_device->getMemoryAllocation( (void*)*address, 
								executive::Device::AnyAllocation );
						if( allocation == 0 )
						{
							globalMemoryError( _device, _dim,
								thread, *address, e.memory_size, e, _kernel );
						}
						_cache.base = ( ir::PTXU64 ) allocation->pointer();
						_cache.extent = allocation->size();
						if( *address >= _cache.base + _cache.extent )
						{
							globalMemoryError( _device, _dim,
								thread, *address, e.memory_size, e, _kernel );
						}
					}
					++address;
				}
				
				break;
			}
			case ir::PTXInstruction::Global:
			{
				TraceEvent::U64Vector::const_iterator 
					address = e.memory_addresses.begin();

				unsigned int threads = e.active.size();
				for( unsigned int thread = 0; thread < threads; ++thread )
				{
					if( !e.active[ thread ] ) continue;
					if( _cache.base > *address 
						|| *address >= _cache.base + _cache.extent
						|| !_cache.valid )
					{
						const executive::Device::MemoryAllocation* allocation = 
							_device->getMemoryAllocation( (void*)*address, 
								executive::Device::AnyAllocation );
						if( allocation == 0 )
						{
							globalMemoryError( _device, _dim,
								thread, *address, e.memory_size, e, _kernel );
						}
						_cache.base = ( ir::PTXU64 ) allocation->pointer();
						_cache.extent = allocation->size();
						if( *address >= _cache.base + _cache.extent )
						{
							globalMemoryError( _device, _dim,
								thread, *address, e.memory_size, e, _kernel );
						}
					}
					++address;
				}
				break;
			}
			case ir::PTXInstruction::Local: 
			{
				bool isGlobalLocal =
					(e.instruction->opcode == ir::PTXInstruction::Ld
					&& e.instruction->a.isGlobalLocal
					&& e.instruction->a.addressMode == ir::PTXOperand::Address)
					|| (e.instruction->opcode == ir::PTXInstruction::St
					&& e.instruction->d.isGlobalLocal
					&& e.instruction->d.addressMode == ir::PTXOperand::Address);
				
				if( isGlobalLocal )
				{
					checkLocalAccess( "GlobalLocal", _dim,
						_globalLocal.base, _globalLocal.extent,
						e, _kernel );
				}
				else
				{
					checkLocalAccess( "Local", _dim,
						_local.base, _local.extent, e, _kernel );
				}
				break;
			}
			case ir::PTXInstruction::Param:
			{
				bool isArgument =
					(e.instruction->opcode == ir::PTXInstruction::Ld
					&& e.instruction->a.isArgument) ||
					(e.instruction->opcode == ir::PTXInstruction::St
					&& e.instruction->d.isArgument);
			
				if( isArgument )
				{
					checkLocalAccess( "Argument", _dim, 
						0, _kernel->getCurrentFrameArgumentMemorySize(),
						e, _kernel );
				}
				else
				{
					checkLocalAccess( "Parameter", _dim, 
						0, _kernel->getCurrentFrameParameterMemorySize(),
						e, _kernel );
				}
				break;
			}
			case ir::PTXInstruction::Shared: checkLocalAccess( "Shared", _dim, 	
				_shared.base, _shared.extent, e, _kernel ); break;
			case ir::PTXInstruction::Const: checkLocalAccess( "Constant", _dim, 
				_constant.base, _constant.extent, e, _kernel ); break;
			default: break;
		}
	}
	
	MemoryChecker::ShadowMemory::ShadowMemory( )
	{
	}

	void MemoryChecker::ShadowMemory::resize( unsigned int size )
	{
		map.resize(size, NOT_DEFINED);
	}
	
	unsigned int MemoryChecker::ShadowMemory::size()
	{
		return map.size();
	} 

	MemoryChecker::Status MemoryChecker::ShadowMemory::checkRegion( 
		const unsigned int idx, const unsigned int size)
	{
		Status currStatus = DEFINED;
		for( unsigned int i=0; i < size; i++ )
		{
			if( currStatus > map[idx+i] )
				currStatus = map[idx+i];
		}
		
		return currStatus;
	}

	void MemoryChecker::ShadowMemory::setRegion( const unsigned int idx, 
		const unsigned int size, const Status stat )
	{
		if( idx+size-1 >= map.size() )
			report( "Store address out of range by " << idx+size-1-map.size()  
				<< "\n" );
		for( unsigned int i=0; i < size; i++ )
		{
				map[idx+i] = stat;
		}
		
	}

	MemoryChecker::Status MemoryChecker::ShadowMemory::checkRegister(
		const ir::PTXOperand::RegisterType idx )
	{
		if( map[ (unsigned int) idx ] < DEFINED ) {
			return NOT_DEFINED;
		}
		return DEFINED;
	}

	void MemoryChecker::ShadowMemory::setRegister(
		const ir::PTXOperand::RegisterType idx, const Status stat )
	{
		assert( (unsigned int)idx < map.size() );
		map[ (unsigned int) idx ] = stat;
	}
	
	void setRegisterStatus( MemoryChecker::ShadowMemory &registerFile, 
		const ir::PTXInstruction &inst, unsigned int regOffset, 
		MemoryChecker::Status stat )
	{
		int regDIdx;
		if( inst.opcode == ir::PTXInstruction::Tex )
		{
			for( int i=0; i < 4; i++ )
			{
				regDIdx = inst.d.array[i].reg + regOffset;
				registerFile.setRegister( regDIdx, stat );
			}
		}
		else
		{
			regDIdx = inst.d.reg + regOffset;
			registerFile.setRegister( regDIdx, stat );
		}
	}

	MemoryChecker::Status MemoryChecker::checkInstruction( 
		const TraceEvent& e,
		bool useMemoryFlag, 
		MemoryChecker::ShadowMemory *shadowMem )
	{
		MemoryChecker::Status destStatus = MemoryChecker::DEFINED;
		TraceEvent::U64Vector::const_iterator 
			address = e.memory_addresses.begin();
			
		unsigned int threads = e.active.size();
		unsigned int regPerThread = _registerFileShadow.size()/threads;	
		const ir::PTXInstruction inst = *(e.instruction);
		std::stringstream errorOut;

		std::string space;
		switch( e.instruction->addressSpace )
		{
			case ir::PTXInstruction::Shared:
				space = "Shared";
				break;
			case ir::PTXInstruction::Local:
				space = "Local";
				break;
			default:
				break;
		}
		
		for ( unsigned int thread = 0; thread < threads; ++thread )
		{
			if ( !e.active[ thread ] ) continue;
		
			errorOut << "Undefined register:";
			ir::PTXOperand d = inst.d;
			unsigned int regD = d.reg+thread*regPerThread;
			destStatus = MemoryChecker::DEFINED;
	
			//check predicate register
			if( inst.pg.reg != ir::PTXOperand::Invalid )
			{
				MemoryChecker::Status varStatus = 
					_registerFileShadow.checkRegister(inst.pg.reg);
				if( varStatus < MemoryChecker::NOT_DEFINED )
				{
					destStatus = MemoryChecker::NOT_DEFINED;
					errorOut << "pg r" 
						<< inst.pg.reg << " ";
				}
			}

			//check register a b c
			const ir::PTXOperand * operands[] = { &inst.a, &inst.b, &inst.c };
			for ( unsigned int i=0; i < 3; i++ )
			{
				if( (operands[i]->addressMode == ir::PTXOperand::Register
					|| operands[i]->addressMode == ir::PTXOperand::Indirect) 
					&& operands[i]->reg != ir::PTXOperand::Invalid )
				{
					int regIdx = operands[i]->reg + thread * regPerThread;
					MemoryChecker::Status varStatus = 
						_registerFileShadow.checkRegister(regIdx);
				
					if( varStatus < MemoryChecker::DEFINED )
					{
						destStatus = MemoryChecker::NOT_DEFINED;
						errorOut << "r" 
							<< operands[i]->reg << " ";
					}
				}
			}	
		
			for( unsigned int i=0; i < inst.c.array.size(); i++ )
			{
				int regIdx = inst.c.array[i].reg + thread * regPerThread;
				MemoryChecker::Status varStatus = 
					_registerFileShadow.checkRegister(regIdx);

				if( varStatus < MemoryChecker::DEFINED )
				{
					destStatus = MemoryChecker::NOT_DEFINED;
					errorOut << "r" 
						<< inst.c.array[i].reg << " ";
				}
			}

			//exception for XOR Rx Ry Ry; Rx always defined
			if( inst.opcode == ir::PTXInstruction::Xor 
				&& inst.a.reg == inst.b.reg )
				return MemoryChecker::DEFINED;
			
			//check memory
			if( ( inst.opcode == ir::PTXInstruction::Ld ||
				inst.opcode == ir::PTXInstruction::Ldu ) &&
				useMemoryFlag )
			{
				MemoryChecker::Status varStatus = 
					shadowMem->checkRegion(*address, e.memory_size);
				++address;
				if( varStatus < MemoryChecker::DEFINED )
				{
					errorOut << "[thread: " << thread 
						<< "] Loading uninitialized value from " << space << 
						" address space" << "Near " << _kernel->location( e.PC ) 
						<< "\n";
					destStatus = MemoryChecker::NOT_DEFINED;
				}
			}

			//store?
			if( inst.opcode == ir::PTXInstruction::St && useMemoryFlag )
			{
				unsigned int pmIndex = *address;		
				unsigned int regIdx = inst.a.reg+thread*regPerThread;
			
				MemoryChecker::Status varStatus = 
					_registerFileShadow.checkRegister(regIdx);
				shadowMem->setRegion(pmIndex, e.memory_size, varStatus);
				++address;

				if( varStatus < MemoryChecker::DEFINED )
				{
					errorOut << "[thread: " << thread 
						<< "] Storing uninitialized value to " << space 
						<< " address space near " << "Near "
						<< _kernel->location( e.PC ) 
						<< "\n";
				}
			}

			if( inst.opcode == ir::PTXInstruction::St 
				&& destStatus == MemoryChecker::NOT_DEFINED
				&& inst.addressSpace == ir::PTXInstruction::Global )
			{
				memoryUninitError( "Global", _dim, _device,
					thread, *address, e.memory_size, e, _kernel );
			}
			
			//store status of destination register: pg, d, or vector d
			if( inst.d.reg == ir::PTXOperand::Invalid )
			{
				if( inst.pq.addressMode == ir::PTXOperand::Invalid )
				{
					destStatus = (destStatus == MemoryChecker::NOT_DEFINED)
						? MemoryChecker::INVALID : destStatus;
					_registerFileShadow.setRegister(inst.pq.reg, destStatus);
				}
			} 
			else if( inst.d.vec != ir::PTXOperand::v1 )
			{
				for( unsigned int i=0; i < inst.d.array.size(); i++ )
				{
					_registerFileShadow.setRegister(
						inst.d.array[i].reg+thread*regPerThread, destStatus);
				}
			} else {
				_registerFileShadow.setRegister(regD, destStatus);
			}
			
			if( destStatus != MemoryChecker::DEFINED )
			{
			  
				report( prefix( thread, _dim, e ) << errorOut.str() << "\n" );
			}
			
			errorOut.str("");
		}
		return destStatus;
	}


	void MemoryChecker::_checkInitialized( const TraceEvent& e )
	{
		switch( e.instruction->addressSpace )
		{
			case ir::PTXInstruction::Shared:
				checkInstruction( e, true, &_sharedShadow );
				break;
			case ir::PTXInstruction::Local:
				checkInstruction( e, true, &_localShadow );
				break;
			default: 	//global, constant, texture
			{
				if( e.instruction->opcode == ir::PTXInstruction::Atom ||
					e.instruction->opcode == ir::PTXInstruction::Ld ||
					e.instruction->opcode == ir::PTXInstruction::Ldu )
				{	
					checkInstruction( e );
				}
				else if( e.instruction->opcode == ir::PTXInstruction::St )
				{	
					if( checkInstruction( e ) == NOT_DEFINED )
						assert(false);
				}
				break;
			}
		}
	}
	
	void MemoryChecker::_checkInstructions( const TraceEvent& e )
	{
		bool errorFlag;	
		switch( e.instruction->opcode )
		{
			case ir::PTXInstruction::Bra:
			case ir::PTXInstruction::Call:
				errorFlag = true;
				break;
			default: 
				errorFlag = false;
				break;
		}
		
		if ( checkInstruction( e ) == NOT_DEFINED && errorFlag )
			uninitError( _dim, 0, e, _kernel );
	}
	
	MemoryChecker::MemoryChecker() : _cache( false ),
		_shared( true ), _local( true ), _constant( true )
	{
	
	}

	void MemoryChecker::setCheckInitialization(bool toggle)
	{
		checkInitialization = toggle;
	}
	
	void MemoryChecker::initialize( const executive::ExecutableKernel& kernel )
	{
		_dim = kernel.blockDim();
	
		_device = kernel.device;
		
		_cache.valid = false;

		_constant.base = 0;
		_constant.extent = kernel.constMemorySize();
		
		_shared.base = 0;
		_shared.extent = kernel.totalSharedMemorySize();
		
		_local.base = 0;
		_local.extent = kernel.localMemorySize();
		
		_globalLocal.base = 0;
		_globalLocal.extent = kernel.globalLocalMemorySize();
		
		_kernel = static_cast< const executive::EmulatedKernel* >( &kernel );

		ir::Dim3 blockDim = kernel.blockDim();
		int threadNum = blockDim.x * blockDim.y * blockDim.z;
		_registerFileShadow.resize(kernel.registerCount() * threadNum);
		
		_sharedShadow.resize(_shared.extent);
		_constShadow.resize(_constant.extent);
		_localShadow.resize(_local.extent);

	}

	void MemoryChecker::event( const TraceEvent& event )
	{
		const bool isMemoryOperation = 
			event.instruction->opcode == ir::PTXInstruction::Ld
			|| event.instruction->opcode == ir::PTXInstruction::Ldu
			|| event.instruction->opcode == ir::PTXInstruction::St
			|| event.instruction->opcode == ir::PTXInstruction::Atom;

		//report( "[" << event.PC << "] " << event.instruction->toString() 
		//	<< "\n" );
			
		if( isMemoryOperation ) 
		{
			_checkAlignment( event );
			_checkValidity( event );
			if( checkInitialization )
				_checkInitialized( event );
		}
		else 
		{
			if( checkInitialization )
				_checkInstructions( event );
		}
	}

	void MemoryChecker::postEvent( const TraceEvent& event )
	{
		if( event.instruction->opcode == ir::PTXInstruction::Call
			|| event.instruction->opcode == ir::PTXInstruction::Ret )
		{
			_local.extent = _kernel->getCurrentFrameLocalMemorySize();
		}
	}
	
	void MemoryChecker::finish()
	{

	}
}

#endif

