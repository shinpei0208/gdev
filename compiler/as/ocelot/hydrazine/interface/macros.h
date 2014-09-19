/*!	\file macros.h
*
*	\brief Header file for common preprocessor macros
*
*	\author Gregory Diamos
*
*	\date : 9/27/2007
*
*/

#ifndef MACROS_H_INCLUDED
#define MACROS_H_INCLUDED

/*!
	\def MAX(a,b)
	\brief a MACRO that returns the max of two numbers
	\param a an untyped argument
	\param b an untyped argument
*/

#define MAX(a,b) (((a)<(b))?(b):(a))


/*!
	\def MIN(a,b)
	\brief a MACRO that returns the max of two numbers
*/

#define MIN(a,b) (((a)<(b))?(a):(b))


/*!
	\def CEIL_DIV(a,b)
	\brief Do a divide of a over b but round up instead of down

*/
#define CEIL_DIV(a,b) (((a) == ((a)/(b)) * (b)) ?  ((a)/(b)):((a)/(b)+1) )


/*!

	\brief Fast swap using XOR

*/
#define SWAP(a,b) \
{\
	(a) ^= (b);\
	(b) ^= (a);\
	(a) ^= (b);\
}

/*!

	\brief Fast absolute value for ints

*/
#define INT_ABS(x) \
{\
	( x ^ ( x >> ( sizeof( int ) * 8 - 1 ) ) ) - ( x >> ( sizeof( int ) * 8 - 1 ) )\
}

/*!

	\brief Regular ABS

*/
#ifndef ABS
#define ABS(x) ( ( ( x ) < 0 ) ? ( ( -( x ) ) ) : ( x ) )
#endif

#endif

