/*!

	\file ActiveTimer.cpp

	\date Wednesday September 24, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The source file for the ActiveTimer class.

*/

#ifndef ACTIVE_TIMER_CPP_INCLUDED
#define ACTIVE_TIMER_CPP_INCLUDED

#include <hydrazine/interface/ActiveTimer.h>
#include <hydrazine/interface/debug.h>
#include <cassert>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace hydrazine
{

	////////////////////////////////////////////////////////////////////////////
	// ActiveTimer::SharedData

	ActiveTimer::SharedData ActiveTimer::_sharedData;

	ActiveTimer::SharedData::SharedData()
	{
		report( "Creating shared data." );
		thread = 0;
		connections = 0;
		alive = false;
		timer.start();
	}
	
	ActiveTimer::SharedData::~SharedData()
	{
		report( "Destroying shared data." );
		assert( connections == 0 );
	}
	
	void ActiveTimer::SharedData::next()
	{
		mutex.lock();

		//report( " Locked mutex, firing expired timers." );
		for( TimerMap::iterator fi = timers.begin(); fi != timers.end();)
		{
			if( fi->first < timer.seconds() )
			{
				fi->second->_done = true;
				fi->second->fired();
				fi->second->_condition.notify_all();
				report( "  Fired ActiveTimer " << fi->second
					<< " at " << fi->first );
				TimerMap::iterator next = fi;
				++next;
				timers.erase( fi );
				fi = next;
			}
			else
			{
				++fi;
			}
		}
				
		if( timers.empty() )
		{
		
			//report( " No active timers, waiting." );		
			condition.notify_all();
			//report( " Unlocking mutex." );
			mutex.unlock();
		
		}
		else
		{
			//report( " Unlocking mutex, yielding." );
			mutex.unlock();
			boost::this_thread::yield();
		}
		
	}
	
	void* ActiveTimer::SharedData::run( void* argument )
	{
	
		report( "Thread is alive." );
		ActiveTimer::SharedData& data = ActiveTimer::_sharedData;
		boost::unique_lock<boost::mutex> lock(data.mutex);
		report( " Locked mutex" );
		assert( !data.alive );
		assert( argument == 0 );
		data.alive = true;
		report( " Signalled alive" );
		data.condition.notify_all();;
		report( " Signalled creator, waiting for signal." );
		data.condition.wait(lock);
		report( " Signal received, unlocking mutex." );
		lock.unlock();
		
		while( data.connections > 0 )
		{
		
			data.next();
		
		}
		
		assert( data.timers.empty() );
		report( " Thread is dying." );
		data.alive = false;
		data.condition.notify_all();
		report( " Thread is dead." );
		
		return 0;
	}
	////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	// ActiveTimer
	ActiveTimer::ActiveTimer()
	{
		report( "Creating new ActiveTimer " << this );
		boost::unique_lock<boost::mutex> lock( _sharedData.mutex );
		++_sharedData.connections;
		
		if( _sharedData.connections == 1 )
		{
		
			assert( _sharedData.thread == 0 );
			
			_sharedData.thread = new boost::thread( SharedData::run, this );
			report( " Waiting on condition " << this );
			_sharedData.condition.wait( lock );
			report( " Signalling " << this );
			_sharedData.condition.notify_all();
		
		}
		
		assert( _sharedData.alive );
		lock.unlock();
		_done = true;
	}
	
	ActiveTimer::~ActiveTimer()
	{
		report( "Destroying ActiveTimer " << this );
		boost::unique_lock<boost::mutex> lock( _sharedData.mutex );
		assert( _sharedData.alive );
		
		if( !_done )
		{
		
			report("Timer not finished, waiting for signal.");
			_sharedData.condition.wait( lock );
		
		}		
		
		--_sharedData.connections;
		
		if( _sharedData.connections == 0 )
		{
		
			_sharedData.condition.notify_all();
			_sharedData.condition.wait( lock );
			assert( !_sharedData.alive );
			delete _sharedData.thread;
			_sharedData.thread = 0;
			
		}
		
		lock.unlock();
		report( "Destroyed ActiveTimer " << this );
	}

	ActiveTimer::ActiveTimer( const ActiveTimer& timer )
	{
	
		report( "Creating new ActiveTimer " << this << " from " << &timer );
		
		boost::unique_lock<boost::mutex> lock( _sharedData.mutex );
		++_sharedData.connections;
		
		if( _sharedData.connections == 1 )
		{
		
			assert( _sharedData.thread == 0 );
			_sharedData.thread = new boost::thread( SharedData::run, this );
			_sharedData.condition.wait( lock );
			_sharedData.condition.notify_all();
		
		}
		
		assert( _sharedData.alive );
		lock.unlock();
		_done = true;
	
	}
	
	const ActiveTimer& ActiveTimer::operator=( const ActiveTimer& timer )
	{
	
		assert( this == &timer );
		return *this;
	
	}

	void ActiveTimer::start( Timer::Second seconds )
	{
	
		boost::unique_lock<boost::mutex> lock( _sharedData.mutex );
		
		if( _sharedData.timers.empty() )
		{
		
			_sharedData.condition.notify_all();;
		
		}
		
		_done = false;
		seconds += _sharedData.timer.seconds();
		_sharedData.timers.insert( std::make_pair( seconds, this ) );
		
		lock.unlock();
	
	}

	void ActiveTimer::wait()
	{
	
		boost::unique_lock<boost::mutex> lock( _sharedData.mutex );
		
		if( !_done )
		{
			
			_condition.wait( lock );
		
		}
		
		lock.unlock();
	
	}
	////////////////////////////////////////////////////////////////////////////
		
}

#endif

