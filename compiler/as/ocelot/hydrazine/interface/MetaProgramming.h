/*! \file MetaProgramming.h
	\date Thursday May 27, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for template meta programming related classes
*/

#ifndef META_PROGRAMMING_H_INCLUDED
#define META_PROGRAMMING_H_INCLUDED

#include <cstdint>

namespace hydrazine
{

/*! \brief Convert a signed type to an unsigned type */
template<typename T>
class SignedToUnsigned
{
	public:
		typedef T type;
};

template<>
class SignedToUnsigned<int8_t>
{
	public:
		typedef uint8_t type;
};

template<>
class SignedToUnsigned<int16_t>
{
	public:
		typedef uint16_t type;
};

template<>
class SignedToUnsigned<int32_t>
{
	public:
		typedef uint32_t type;
};

template<>
class SignedToUnsigned<int64_t>
{
	public:
		typedef uint64_t type;
};

/*! \brief Determine if an integer type is negative */
template<typename T>
bool isNegative(T t)
{
	return t < 0;
}

inline bool isNegative(unsigned char)
{
	return false;
}

inline bool isNegative(unsigned short)
{
	return false;
}

inline bool isNegative(unsigned int)
{
	return false;
}

inline bool isNegative(long long unsigned int)
{
	return false;
}

}

#endif

