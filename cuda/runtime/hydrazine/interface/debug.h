
/*!	\file debug.h
*
*	\brief Header file for common debug macros
*
*	\author Gregory Diamos
*
*	\date : Sunday November 16, 2008
*
*/


#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <iostream>
#include <sstream>
#include <cassert>
#include <hydrazine/interface/Timer.h>

#include <iosfwd>

namespace hydrazine
{
	/*! \brief Global report timer */
	extern Timer _ReportTimer;

	/*! \brief Return a string representing the current system time */
	extern std::string _debugTime();

	/*! \brief Return a formatted line number and file name. */
	extern std::string _debugFile( const std::string& file, unsigned int line );

	/*! \brief Convert an iterable range to a string
		
		T must implement the concept of a forward iterator and the object being
			pointed to must be able to be accepted by operator<<
		
		\param begin Iterator to the start of the range
		\param end Iterator to the end of the range
		\param space The string used as a space
		
		\return A string representation of the specified range.
	*/
	template < typename T > std::string toString( T begin, T end, 
		std::string space = " ", unsigned int limit = 80 )
	{
		std::stringstream stream;
		
		if( begin != end )
		{
			stream << *begin;
			++begin;
		}
		
		for( T iterator = begin; iterator != end; ++iterator )
		{
			stream << space;			
			stream << *iterator;
			if( stream.str().size() > limit )
			{
				break;
			}
		}
		
		return stream.str();
	}

	/*! \brief Convert an iterable range to a string using a formatting functor
		
		T must implement the concept of a forward iterator and the object being
			pointed to must be able to be accepted 
			by operator<< format( object )
		
		\param begin Iterator to the start of the range
		\param end Iterator to the end of the range
		\param space The string used as a space
		
		\return A string representation of the specified range.

	*/
	template < typename T, typename Format > std::string toFormattedString( 
		T begin, T end, Format format, std::string space = " ", 
		unsigned int limit = 80 )
	{
		std::stringstream stream;
		
		if( begin != end )
		{
			stream << format( begin );
			++begin;
		}
		
		for( T iterator = begin; iterator != end; ++iterator )
		{
			stream << space;			
			stream << format( iterator );
			if( stream.str().size() > limit )
			{
				break;
			}
		}
		
		return stream.str();
	}

	/*!
		\brief Strip the front of a file	
	*/
	template< char delimiter >
	std::string stripReportPath( const std::string& string )
	{
		size_t found = string.find_last_of(delimiter);
		std::string result = string.substr(found+1);
		return result;
	}

	struct NullStream : std::ostream {};

	static NullStream nullstream;
	
	/*! \brief Return the stream with the current name */
	extern std::ostream& _getStream(const std::string& name);
	
	#if 0
	inline std::ostream& log(const std::string& path)
	{
			return nullstream;
	}
	#else
	inline std::ostream& log(const std::string& path)
	{
			return _getStream(path);
	}		
	#endif
	
	extern void enableAllLogs();
	
}

// Swallow all types
template <typename T>
hydrazine::NullStream & operator<<(hydrazine::NullStream & s, T const &)
{
	return s;
}

// Swallow manipulator templates
inline hydrazine::NullStream & operator<<(hydrazine::NullStream & s,
	std::ostream &(std::ostream&))
{
	return s;
}

/*!

	\def REPORT_ERROR_LEVEL
	
	\brief The threshold to print out the debugging message.
	
	If the debugging error levels is less than this, it will not be printed out.

*/

#ifndef REPORT_ERROR_LEVEL
#define REPORT_ERROR_LEVEL 1
#endif

/*!
	\def reportE(x,y)
	\brief a MACRO that prints a string to stdio if DEBUG is defined and x is 
	greater than REPORT_ERROR_LEVEL, or exits the program if the error level 
	is greater than EXIT_ERROR_LEVEL.  
	
	If MPI_DEBUG is defined, it appends the rank to the beginning of the error 
	message.
	
	\param x The error level
	\param y The message to print.  You can use the << operators to send 
		multiple arguments
*/

#ifndef NDEBUG
	#define reportE(x, y) \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL && (x) >= REPORT_ERROR_LEVEL)\
		{ \
			{\
			std::cout << "(" << hydrazine::_debugTime() << ") " \
				<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
				<< " " << y << "\n";\
			}\
		 \
		}
#else
	#define reportE(x, y)
#endif

/*!
	\def report(a)
	\brief a MACRO that prints a string to stdio if DEBUG is defined
	\param a a string
*/

#ifndef NDEBUG
	#define report(y) \
		if(REPORT_BASE >= REPORT_ERROR_LEVEL)\
		{ \
			{\
			std::cout << "(" << hydrazine::_debugTime() << ") " \
				<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
					<< " " << y << "\n";\
			}\
		 \
		}
#else
	#define report(y)
#endif

/*! \brief An assertion with a message */
#ifndef NDEBUG

	#define assertM(x,y) \
		if(!(x))\
		{ \
			{\
			std::cout << "(" << hydrazine::_debugTime() << ") " \
				<< hydrazine::_debugFile( __FILE__, __LINE__ ) \
					<< " Assertion message: " << y << "\n";\
			}\
			assert(x);\
		 \
		}
#else
	#define assertM(x,y)
#endif

#endif

