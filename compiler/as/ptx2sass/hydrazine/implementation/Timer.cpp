/*!	\file Timer.cpp
*
*	\brief Source file for the Timer class
*
*	Author: Gregory Diamos
*
*
*/

#ifndef TIMER_CPP_INCLUDED
#define TIMER_CPP_INCLUDED

#include <hydrazine/interface/Timer.h>
#include <sstream>

namespace hydrazine
{
	std::string Timer::toString() const
	{
		std::stringstream stream;
		
		#ifdef HAVE_TIME_H
			stream << seconds() << "s (" << cycles() << " ns)";
		#else
			stream << seconds() << "s (" << cycles() << " ticks)";
		#endif
		
		return stream.str();
	}
}

#endif

