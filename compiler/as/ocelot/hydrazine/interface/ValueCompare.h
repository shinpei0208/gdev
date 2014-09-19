/*!
	\file ValueCompare.h
	\date Wednesday May 27, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the ValueCompare class.	
*/
#ifndef VALUE_COMPARE_H_INCLUDED
#define VALUE_COMPARE_H_INCLUDED

#include <functional>

namespace hydrazine
{
	/*!
		\brief A class for comparing key/value pairs based on the key only
	*/
	template< typename Compare, typename Container >
	class ValueCompare : 
		public std::binary_function< typename Container::value_type, 
			typename Container::value_type, bool >, 
		public std::binary_function< typename Container::value_type, 
			typename Container::key_type, bool >, 
		public std::binary_function< typename Container::key_type, 
			typename Container::value_type, bool >
	{
		public:
			typedef size_t size_type;
			typedef typename Container::key_type key_type;
			typedef typename Container::mapped_type mapped_type;
			typedef typename Container::value_type value_type;
			typedef ptrdiff_t difference_type;
			typedef value_type* pointer;
			typedef value_type& reference;
			typedef const value_type* const_pointer;
			typedef const value_type& const_reference;
			
		protected:
			Compare _compare;
	
		public:
			const Compare& compare() const
			{
				return _compare;
			}
		
		public:
			explicit ValueCompare( const Compare& c ) : _compare( c ) {}
			ValueCompare() {}
	
		public:
			bool operator()( const_reference x, const_reference y ) const
			{
				return _compare( x.first, y.first );
			}

			bool operator()( const key_type& x, const_reference y ) const
			{
				return _compare( x, y.first );
			}

			bool operator()( const_reference x, const key_type& y ) const
			{
				return _compare( x.first, y );
			}
	};

}

#endif

