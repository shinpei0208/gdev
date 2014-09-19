/*!
	\file Iterator.h
	\date Tuesday March 12, 2009
	\author Gregory Diamos
	\brief Header file for Iterator classes and concepts
*/

#ifndef ITERATOR_H_INCLUDED
#define ITERATOR_H_INCLUDED

#include <iterator>

namespace hydrazine
{


	/*!
		\brief Define typedefs for iterator classes
	*/
	template< typename Category, typename T, typename Distance, 
		typename Pointer, typename Reference >
	struct Iterator
	{
		//! Concept
		typedef Category iterator_category;
		//! The type "pointed to" by the iterator.
		typedef T value_type;
		//! Distance between iterators is represented as this type.
		typedef Distance difference_type;
		//! This type represents a pointer-to-value_type.
		typedef Pointer pointer;
		//! This type represents a reference-to-value_type.
		typedef Reference reference;		
	};

	/*!
		\brief A class for specializing pointer and reference types
	*/
	template< typename Pointer >
	struct IteratorTraits
	{
		typedef typename Pointer::iterator_category iterator_category;
		typedef typename Pointer::value_type value_type;
		typedef typename Pointer::difference_type difference_type;
		typedef typename Pointer::pointer pointer;
		typedef typename Pointer::reference reference;
	};

	template< typename T >
	struct IteratorTraits< T* >
	{
		typedef std::random_access_iterator_tag iterator_category;
		typedef T value_type;
		typedef ptrdiff_t difference_type;
		typedef T* pointer;
		typedef T& reference;
	};

	template< typename T >
	struct IteratorTraits< const T* >
	{
		typedef std::random_access_iterator_tag iterator_category;
		typedef T value_type;
		typedef ptrdiff_t difference_type;
		typedef const T* pointer;
		typedef const T& reference;
	};
	
	/*!
		\brief A generic iterator class wrapper for iterators derived from
			pointers.
			
		Container is here so that different iterator types get instantiated
		for containers using the same underlying pointer as in GCC-4.3.3
	*/
	template< typename Type, typename Container >
	class PointerIterator
	{
		protected:
			Type _current;
		
		public:
			typedef Type iterator_type;
			typedef typename IteratorTraits< Type >::iterator_category 
				iterator_category;
			typedef typename IteratorTraits< Type >::value_type value_type;
			typedef typename IteratorTraits< Type >::difference_type 
				difference_type;
			typedef typename IteratorTraits< Type >::pointer pointer;
			typedef typename IteratorTraits< Type >::reference reference;
	
		public:
			PointerIterator() : _current( Type() ) {}
			
			explicit PointerIterator( const Type& it ) : _current( it ) 
			{}
			
			template< typename _Assignee >
			PointerIterator( const PointerIterator< _Assignee, 
				typename Container::type >& it ) : _current( it.base() )
			{
			
			}
			
			reference operator*() const
			{
				return *_current;
			}
			
			pointer operator->() const
			{
				return _current;
			}
			
			PointerIterator& operator++()
			{
				++_current;
				return *this;
			}
			
			PointerIterator operator++( int )
			{
				return PointerIterator( _current++ );
			}
			
			PointerIterator& operator--()
			{
				--_current;
				return *this;
			}
			
			PointerIterator operator--( int )
			{
				return PointerIterator( _current-- );
			}
			
			reference operator[]( const difference_type& n )
			{
				return _current[ n ];
			}
			
			PointerIterator& operator+=( const difference_type& n )
			{
				_current += n;
				return *this;
			}
			
			PointerIterator operator+( const difference_type& n )
			{
				return PointerIterator( _current + n );
			}
			
			PointerIterator& operator-=( const difference_type& n )
			{
				_current -= n;
				return *this;
			}
			
			PointerIterator operator-( const difference_type& n )
			{
				return PointerIterator( _current - n );
			}
			
			const Type& base() const
			{
				return _current;
			}
			
	};
	
	template<typename TypeL, typename TypeR, typename Container>
	inline bool operator==(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{ 
		return left.base() == right.base(); 
	}

	template<typename Pointer, typename Container>
	inline bool operator==(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() == right.base(); 
	}

	template<typename TypeL, typename TypeR, typename Container>
	inline bool operator!=(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{
		return left.base() != right.base(); 
	}

	template<typename Pointer, typename Container>
	inline bool operator!=(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() != right.base(); 
	}

	template<typename TypeL, typename TypeR, typename Container>
	inline bool operator<(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{
		return left.base() < right.base(); 
	}

	template<typename Pointer, typename Container>
	inline bool operator<(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() < right.base(); 
	}

	template<typename TypeL, typename TypeR, typename Container>
	inline bool operator>(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{
		return left.base() > right.base(); 
	}

	template<typename Pointer, typename Container>
	inline bool operator>(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() > right.base(); 
	}

	template<typename TypeL, typename TypeR, typename Container>
	inline bool operator<=(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{
		return left.base() <= right.base(); 
	}

	template<typename Pointer, typename Container>
	inline bool operator<=(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() <= right.base(); 
	}

	template<typename TypeL, typename TypeR, typename Container>
	inline bool operator>=(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{
		return left.base() >= right.base(); 
	}

	template<typename Pointer, typename Container>
	inline bool operator>=(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() >= right.base(); 
	}

	template<typename TypeL, typename TypeR, typename Container>
	inline typename PointerIterator<TypeL, Container>::difference_type
		operator-(const PointerIterator<TypeL, Container>& left,
		const PointerIterator<TypeR, Container>& right)
	{
		return left.base() - right.base(); 
	}

	template<typename Pointer, typename Container>
	inline typename PointerIterator<Pointer, Container>::difference_type
		operator-(const PointerIterator<Pointer, Container>& left,
		const PointerIterator<Pointer, Container>& right)
	{
		return left.base() - right.base(); 
	}

	template<typename Pointer, typename Container>
	inline PointerIterator<Pointer, Container>
		operator+(typename PointerIterator<Pointer, Container>::difference_type
		n, const PointerIterator<Pointer, Container>& it )
	{
		return PointerIterator<Pointer, Container>( it.base() + n ); 
	}

}

#endif

