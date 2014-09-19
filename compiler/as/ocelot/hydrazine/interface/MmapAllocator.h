/*!
	\file MmapAllocator.h
	\date Wednesday May 13, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the MmapAllocator class
*/

#ifndef MMAP_ALLOCATOR_H_INCLUDED
#define MMAP_ALLOCATOR_H_INCLUDED

#include <new>
#include <sys/mman.h>

namespace hydrazine
{
	/*!
		\brief An allocator that draws from blocks of file backed memory
	*/
	template< typename T >
	class MmapAllocator
	{
		public:
			typedef size_t size_type;
			typedef ptrdiff_t difference_type;
			typedef T* pointer;
			typedef const T* const_pointer;
			typedef T& reference;
			typedef const T& const_reference;
			typedef T value_type;
			
		public:
			template< typename NewT >
			struct rebind
			{
				typedef MmapAllocator< NewT > other;
			};
			
			MmapAllocator() throw() {}
			MmapAllocator( const MmapAllocator& ) throw() {}
			
			template< typename SomeT >
			MmapAllocator( const MmapAllocator< SomeT >& ) throw() {}
			
			~MmapAllocator() throw() {}
			
			pointer address( reference r ) { return &r; }
			const_pointer address( const_reference r ) { return &r; }
			
			pointer allocate( size_type n, const void* = 0 )
			{
				if( n > max_size() )
				{
					throw std::bad_alloc();
				}
				return static_cast< pointer >( mmap( 0, 
					n * sizeof( value_type ), 
					PROT_READ | PROT_WRITE | PROT_EXEC, 
					MAP_PRIVATE | MAP_ANONYMOUS, 0, 0 ) ); 
			}
			
			void deallocate( pointer p, size_type s )
			{
				munmap( p, s * sizeof( value_type ) );
			}
			
			size_type max_size() const throw()
			{
				return size_type( -1 ) / sizeof( value_type );
			}
			
			void construct( pointer p, const_reference val )
			{
				::new(p) value_type( val );
			}
			
			void destroy( pointer p )
			{
				p->~value_type();
			}
			
	};

	template< typename T >
	inline bool operator==( const MmapAllocator< T >&, 
		const MmapAllocator< T >& )
	{
		return true;
	}

	template< typename T >
	inline bool operator!=( const MmapAllocator< T >&, 
		const MmapAllocator< T >& )
	{
		return false;
	}

}

#endif

