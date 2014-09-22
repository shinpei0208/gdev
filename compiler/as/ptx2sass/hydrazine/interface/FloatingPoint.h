/*! \file   FloatingPoint.h
	\date   Friday August 26, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for platform independent floating point modifiers.
*/

#ifndef FLOATING_POINT_H_INCLUDED
#define FLOATING_POINT_H_INCLUDED

#ifndef _WIN32
#include <cfenv>
#else

#define FE_TONEAREST  0
#define FE_TOWARDZERO 0
#define FE_DOWNWARD   0
#define FE_UPWARD     0

#endif


namespace hydrazine
{
	/*! \brief Get the current floating point rounding mode */
	int fegetround();
	/*! \brief Set the current floating point rounding mode */
	int fesetround(int value);
	
	/*! \brief Copy the sign */
	float copysign(float value, float sign);
	
	/*! \brief Is the value not a number? */
	bool isnan(float value);
	/*! \brief Is the value infinite? */
	bool isinf(float value);
	/*! \brief Is the value normal? */
	bool isnormal(float value);

	/*! \brief Round to nearest int */
	float nearbyintf(float value);
	/*! \brief Round to nearest int */
	double nearbyintf(double value);
	/*! \brief Round to negative infinity */
	float trunc(float value);
	/*! \brief Round to negative infinity */
	double trunc(double value);

	/*! \brief two to the power of */
	float exp2f(float value);
	/*! \brief log base 2 */
	float log2f(float value);
}

#endif

