/*!	\file math.h
*
*	\brief Header file for common math functions.
*
*	\author Gregory Diamos
*
*	\date : 9/27/2007
*	\date : 10/24/2007 : added exceptions
*	\date : 5/27/2010 : added extended precision multiply
*   \date : 8/26/2011 : added fenv
*
*/

#ifndef MATH_H_INCLUDED
#define MATH_H_INCLUDED

#include <hydrazine/interface/MetaProgramming.h>
#include <cassert>
#include <algorithm>

/*! \brief a namespace for common classes and functions */
namespace hydrazine
{

	/*! \brief Check to see if an int is a power of two */
	inline bool isPowerOfTwo (int value);

	/*! \brief Check to see if an unsigned int is a power of two */
	inline bool isPowerOfTwo (unsigned int value);

	/*! \brief Mod a power of power of two */
	inline unsigned int modPowerOfTwo( unsigned int value1, 
		unsigned int value );

	/*! \brief Mod a power of power of two */
	inline int modPowerOfTwo( int value1, int value );
		
	/*! \brief Compute the next highest power of two */
	inline unsigned int powerOfTwo( unsigned int value );

	/*! \brief Modes for permute instructions */
	enum PermuteMode
	{
		DefaultPermute,
		ForwardFourExtract,
		BackwardFourExtract,
		ReplicateEight,
		EdgeClampLeft,
		EdgeClampRight,
		ReplicateSixteen
	};

	/*! \brief Read a byte from a 64-bit word depending on the mode */
	template< PermuteMode mode >
	unsigned int readBytes( long long unsigned int value, unsigned int control, 
		unsigned int byte );

	/*! \brief Permute bytes from a pair of 32-bit operands */
	template< PermuteMode mode >
	unsigned int permute( unsigned int a, unsigned int b, 
		unsigned int control );

	/*! \brief Insert a bit-field in an operand */
	template< typename type >
	type bitFieldInsert( type a, type b, unsigned int c, unsigned int d);

	/*! \brief Reverse the bits in the operand */
	template< typename type >
	type brev( type value );

	/*! \brief Get a bit at a single position */
	template< typename type >
	type bitExtract( type value, unsigned int position );

	/*! \brief Set a bit at a single position */
	template< typename type >
	type bitInsert( type value, unsigned int position );

	/*! \brief Extract a bit field */
	template< typename type >
	type bfe( type value, unsigned int position, unsigned int length, bool isSigned );

    /*! \brief Compute the number of bits set in the operand */
	template< typename type >
	unsigned int bfind( type value, bool shiftAmount );
	
    /*! \brief Compute the number of bits set in the operand */
	template< typename type >
	unsigned int popc( type value );

	/*! \brief Compute the number of leading zeros in the operand */
	template< typename type >
	unsigned int countLeadingZeros( type value );
	
	/*! \brief Perform extended precision multiply */
	template< typename type >
	void multiplyHiLo( type& hi, type& lo, type r0, type r1 );

	/*! \brief Perform extended precision add */
	template< typename type >
	void addHiLo( type& hi, type& lo, type r0 );
	
	////////////////////////////////////////////////////////////////////////////
	// Power of two checks
	inline bool isPowerOfTwo( int value )
	{
		return (value & (~value+1)) == value;
	}
	
	inline bool isPowerOfTwo( unsigned int value )
	{
		return (value & (~value+1)) == value;
	}
	
	inline unsigned int modPowerOfTwo( unsigned int value1, unsigned int value )
	{
		assert( value != 0 );
		return value1 & ( value - 1 );
	}

	inline int modPowerOfTwo( int value1, int value )
	{
		assert( value != 0 );
		return value1 & ( value - 1 );
	}
	
	inline unsigned int nextPowerOfTwo( unsigned int value )
	{	
		value--;		
		value |= value >> 1;		
		value |= value >> 2;		
		value |= value >> 4;		
		value |= value >> 8;		
		value |= value >> 16;		
		value++;		
		return value;		
	}
	////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	// Bit Manipulation
	template< typename type >
	unsigned int countLeadingZeros( type value )
	{
		unsigned int count = 0;
		unsigned int max = 8 * sizeof( type );
		type mask = ( ( type ) 1 ) << ( max - 1 );
		
		while( count < max && ( ( value & mask ) == 0 ) )
		{
			++count;
			value <<= 1;
		}
		
		return count;	
	}

	template< typename type >
	unsigned int popc( type value )
	{
		unsigned int count = 0;
		
		while( value != 0 )
		{
			if( value & 1 ) ++count;
			value >>= 1;
		}
		
		return count;	
	}

	template< typename type >
	class InvertIfNegative
	{
	public:
		type invert(const type& t) { return t; }
	};

	template< >
	class InvertIfNegative<int32_t>
	{
	public:
		int32_t invert(const int32_t& t) { return t < 0 ? -t : t; }
	};

	template< >
	class InvertIfNegative<int64_t>
	{
	public:
		int64_t invert(const int64_t& t) { return t < 0 ? -t : t; }
	};

	template< typename type >
	unsigned int bfind( type value, bool shiftAmount )
	{
		unsigned int d = -1;
		int64_t msb = 8 * sizeof( type ) - 1;
		
		InvertIfNegative< type > inverter;
		
		value = inverter.invert( value );
		
		for( int64_t i = msb; i >= 0; --i )
		{
			if( value & ( 1ULL << i ) )
			{
				d = i;
				break;
			}
		}
		
		if( shiftAmount )
		{
			if( d != ( unsigned int ) - 1 )
			{
				d = msb - d;
			}
		}

		return d;
	}

	template< typename type >
	type bitExtract( type value, unsigned int position )
	{
		value >>= position;
		value &= 1;
		return value;
	}

	template< typename type >
	type bitInsert( type value, type bit, unsigned int position )
	{
		bit &= 1;
		bit <<= position;
		type mask = ~((type)1 << position);
		value &= mask;
		value |= bit;
		return value;
	}

	template< typename type >
	type brev( type value )
	{
		type msb = sizeof( type ) * 8 - 1;
		
		type result = 0;
		for( unsigned int i = 0; i <= msb; ++i )
		{
			result = bitInsert( result, bitExtract( value, msb - i ), i );
		}
		
		return result;
	}
	
    template< typename type >
    type bfe( type value , unsigned int pos, unsigned int len, bool isSigned)
    {
      pos = pos & 0xff;
      len = len & 0xff;
      type result = 0;
      unsigned int msb = sizeof( type ) * 8 - 1;
      unsigned int spos = ((pos + len - 1) > msb) ? msb : pos + len - 1;
      type sbit = (!isSigned || (len==0)) ? 0 : bitExtract(value,spos);

      for(unsigned int i = 0 ; i<=msb; ++i)
      {
        if(i < len && (pos + i)<=msb) 
        {
          result = bitInsert(result, bitExtract(value,pos + i),i);
        }
        else
        {
          result = bitInsert(result, sbit, i); 
        }
      }

      return result;
    }

	template< typename type >
	type bitFieldInsert( type a, type b, 
		unsigned int position, unsigned int length)
	{
		unsigned int msb = sizeof( type ) * 8 - 1;
		
		type result = b;
		
		for( unsigned int i = 0; i < length && ( i + position ) <= msb; ++i )
		{
			result = bitInsert( result, bitExtract( a, i ), i + position );
		}
		
		return result;
	}

	template< PermuteMode mode >
	unsigned int readByte( long long unsigned int value, unsigned int control, 
		unsigned int byte )
	{
		long long unsigned int result = value;
		switch( mode )
		{
			case DefaultPermute:
			{
				result >>= (control & 0x7) * 8;
				if((control >> 3) & 1)
				{
					if(((result & 0xff) >> 7) & 1)
					{
						result = 0xff;
					}
					else
					{
						result = 0x0;
					}					
				}
				break;
			}
			case ForwardFourExtract:
			{
				result >>= ( control + byte ) * 8;
				break;
			}
			case BackwardFourExtract:
			{
				result >>= ( ( 8 + control - byte ) & 0x7 ) * 8;
				break;
			}
			case ReplicateEight:
			{
				result >>= control * 8;
				break;
			}
			case EdgeClampLeft:
			{
				result >>= std::max( control, byte ) * 8;
				break;
			}
			case EdgeClampRight:
			{
				result >>= std::min( control, byte ) * 8;
				break;
			}
			case ReplicateSixteen:		
			{
				result >>= (( byte & 0x1 ) + ( ( control & 0x1 ) << 1 )) * 8;
				break;
			}
		}
		return result & 0xff;
	}

	template< PermuteMode mode >
	unsigned int permute( unsigned int a, unsigned int b, unsigned int control )
	{
		long long unsigned int extended = 
			( ( ( long long unsigned int ) b ) << 32 ) 
			| ( long long unsigned ) a;
		
		if( mode == DefaultPermute )
		{
			unsigned int control0 = (control >> 0) & 0xf;
			unsigned int control1 = (control >> 4) & 0xf;
			unsigned int control2 = (control >> 8) & 0xf;
			unsigned int control3 = (control >> 12) & 0xf;
			
			unsigned int result = readByte< mode >( extended, control0, 0 );
			result |= readByte< mode >( extended, control1, 1 ) << 8;
			result |= readByte< mode >( extended, control2, 2 ) << 16;
			result |= readByte< mode >( extended, control3, 3 ) << 24;
			
			return result;
		}
		else
		{
			control &= 0x3;
			
			unsigned int result = readByte< mode >( extended, control, 0 );
			result |= readByte< mode >( extended, control, 1 ) << 8;
			result |= readByte< mode >( extended, control, 2 ) << 16;
			result |= readByte< mode >( extended, control, 3 ) << 24;
			
			return result;
		}
	}
	////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	// Extended precision
	template< typename type >
	void multiplyHiLo( type& hi, type& lo, type r0, type r1 )
	{
		bool r0Signed = isNegative( r0 );
		bool r1Signed = isNegative( r1 );

		r0 = ( r0Signed ) ? -r0 : r0;
		r1 = ( r1Signed ) ? -r1 : r1;
	
		typedef typename SignedToUnsigned< type >::type utype;

		bool negative = ( r0Signed && !r1Signed ) || ( r1Signed && !r0Signed );

		unsigned int bits = sizeof( type ) * 4;
		utype mask = ( ( type ) 1 << bits ) - 1;
	
		// A B
		// C D
		//
		//    DA DB
		// CA CB

		utype a = ( utype ) r0 >> bits;
		utype b = ( utype ) r0 & mask;
		utype c = ( utype ) r1 >> bits;
		utype d = ( utype ) r1 & mask;

		// 

		utype da = d * a;
		utype db = d * b;
		utype ca = c * a;
		utype cb = c * b;

		utype totalBits = ( sizeof( type ) * 8 - 1);
		utype upperMask = ( ( type ) 1 << totalBits ) - 1;
		utype daUpper = da >> totalBits;
		utype cbUpper = cb >> totalBits;
		utype xCarryIn = ( ( da & upperMask ) 
			+ ( cb & upperMask ) ) >> totalBits;
		utype xCarryOut = ( daUpper + cbUpper + xCarryIn ) >> 1;

		utype x = da + cb;

		utype xUpper = x >> totalBits;
		utype yCarryIn = ( ( x & upperMask ) + ( db >> bits ) ) >> totalBits;
		utype yCarryOut = ( yCarryIn + xUpper ) >> 1;

		utype y = ( db >> bits ) + x;

		lo = ( db & mask ) | ( ( y & mask ) << bits );
		hi = ( y >> bits ) + ca + ( ( yCarryOut + xCarryOut ) << bits );
	
		utype loUpperBit = ( utype )( ~( utype ) lo ) >> totalBits;
		utype signCarryIn = ( ( ( utype ) ( ~( utype ) lo ) 
			& upperMask ) + 1 ) >> totalBits;
		utype signCarryOut = ( loUpperBit + signCarryIn ) >> 1;

		lo = ( negative ) ? -lo : lo;
		hi = ( negative ) ? ( utype )( ~( utype ) hi ) + signCarryOut : hi;
	}
	
	template< typename type >
	void addHiLo( type& hi, type& lo, type r0 )
	{
		typedef typename SignedToUnsigned< type >::type utype;
	
		utype uHi = hi;
		utype uLo = lo;
		utype uR0 = r0;
		
		utype loResult = uR0 + uLo;
		utype carry = (loResult < uR0 || loResult < uLo) ? 1 : 0;
		utype hiResult = uHi + carry;
		
		hi = hiResult;
		lo = loResult;
	}

	template< typename type >
	void add(type& result, type& carry, type r1, type r0, type cIn)
	{
		typedef typename SignedToUnsigned< type >::type utype;
		utype uR0 = r0;
		utype uR1 = r1;
		
		utype loResult = uR0 + uR1 + cIn;
		carry = (loResult < uR0 || loResult < uR1) ? 1 : 0;
		
		result = loResult;
	}
	////////////////////////////////////////////////////////////////////////////

}

#endif

