/*!
*	\file Thread.hpp
*
*	\author Gregory Diamos
*	\date Wednesday November 12, 2008
*	
*
*	
*
*	\brief 	The template source file for the Thread interface;
*/

#ifndef THREAD_HPP_INCLUDED
#define THREAD_HPP_INCLUDED

#include "Thread.h"

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE 
#endif

#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////

namespace hydrazine
{

	template < class T >
	Thread::Id Thread::threadReceive( T*& data, Thread::Id id )
	{

		report( "Thread " << _id << " thread blocking receive." );
	
		Message message = _threadQueue.pull( id );
		assert( message.destination == _id );
		
		data = static_cast< T* >( message.payload );
		
		return message.source;
		
	}


	template < class T >
	void Thread::receive( T*& data )
	{

		report( "Thread " << _id << " controller blocking receive." );

		Message message = _group->pull( THREAD_ANY_ID );
		assert( message.destination == THREAD_CONTROLLER_ID );
		
		data = reinterpret_cast< T* >( message.payload );
			
	}

}

////////////////////////////////////////////////////////////////////////////////
#endif

