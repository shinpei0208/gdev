/*!
	\file Thread.cpp

	\author Gregory Diamos
	\date 4/10/2008

	\brief 	The source file for the thread class
		This file provides a class wrapper for pthreads
*/

#ifndef THREAD_CPP_INCLUDED
#define THREAD_CPP_INCLUDED

#include <hydrazine/interface/Thread.h>

#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Exception.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef REPORT_BASE
#undef REPORT_BASE 
#endif

#define REPORT_BASE 0

namespace hydrazine
{

	Thread::Queue::Queue()
	{
	
	}
	
	Thread::Queue::~Queue()
	{
		assert( _queue.empty() );	
	
	}
	
	void Thread::Queue::push( const Message& message )
	{
		_mutex.lock();
		
		_queue.push_back( message );
		
		_condition.notify_all();
		_mutex.unlock();
	}
	
	Thread::Message Thread::Queue::pull( Id id )
	{
		Message result;
		result.type = Message::Invalid;
		
		// lock is implied
		boost::unique_lock<boost::mutex> lock(_mutex);
		
		while( result.type == Message::Invalid )
		{
			for( MessageQueue::iterator message = _queue.begin(); 
				message != _queue.end(); ++message )
			{
				if( _compare( message->source, id ) )
				{
					assert( message->type != Message::Invalid );
					result = *message;
					_queue.erase( message );
					break;
				}
			}
			
			if( result.type == Message::Invalid )
			{
				_condition.wait( lock );
			}
		}
			
		lock.unlock();
		
		return result;
	}

	bool Thread::Queue::test( Id& id, bool block )
	{
		bool found = false;
	
		// lock is implied
		boost::unique_lock<boost::mutex> lock( _mutex );
		
		do
		{
			for( MessageQueue::iterator message = _queue.begin(); 
					message != _queue.end(); ++message )
			{
				if( _compare( message->source, id ) )
				{
					assert( message->type != Message::Invalid );
					id = message->source;
					found = true;
					break;
				}
			}
			
			if( block && !found )
			{
				_condition.wait( lock );
			}
		}
		while( block && !found );
		
		lock.unlock();
		
		return found;
	}

	Thread::Group::Group()
	{
	
	}

	Thread::Group::~Group()
	{
		assert( empty() );
	}

	void Thread::Group::add( Thread* thread )
	{
		_mutex.lock();
	
		assert( _threads.count( thread->id() ) == 0 );
	
		_threads.insert( std::make_pair( thread->id(), thread ) );
	
		_mutex.unlock();
	}

	void Thread::Group::remove( Thread* thread )
	{
		_mutex.lock();
	
		assert( _threads.count( thread->id() ) != 0 );
	
		_threads.erase( thread->id() );
	
		_mutex.unlock();
	}

	Thread* Thread::Group::find( Id id )
	{
		Thread* result = 0;
	
		_mutex.lock();
	
		ThreadMap::iterator thread = _threads.find( id );
	
		if( thread != _threads.end() )
		{
			result = thread->second;
		}
	
		_mutex.unlock();
	
		return result;	
	}

	void Thread::Group::push( const Message& message )
	{
		assert( message.destination != THREAD_ANY_ID );
	
		if( message.destination == THREAD_CONTROLLER_ID )
		{
			assert( _threads.count( message.source ) != 0 );		
			_controllerQueue.push( message );
		}
		else
		{
			_mutex.lock();

			ThreadMap::iterator thread = _threads.find( message.destination );
			assert( thread != _threads.end() );
		
			thread->second->_threadQueue.push( message );

			_mutex.unlock();
		}
	}
	
	Thread::Message Thread::Group::pull( Id source )
	{
		return _controllerQueue.pull( source );
	}

	bool Thread::Group::test( Id& source, bool block )
	{
		return _controllerQueue.test( source, block );	
	}

	bool Thread::Group::empty() const
	{
		return _threads.empty();
	}

	unsigned int Thread::Group::size() const
	{
		return _threads.size();
	}

	Thread::Id Thread::_nextId = THREAD_START_ID;

	void* Thread::_launch( void* argument )
	{
		Thread* thread = static_cast< Thread* >( argument );	
		thread->execute();
		return 0;
	}

	bool Thread::_compare( Id one, Id two )
	{
		return one == two || one == THREAD_ANY_ID || two == THREAD_ANY_ID;
	}
	
	bool Thread::threadTest( Id id )
	{
		bool null = false;
		return _threadQueue.test( id, null );
	}

	void Thread::threadSend( MessageData data, Thread::Id id )
	{

		assert( id != THREAD_ANY_ID );
		assert( _id != id );

		Message message;
		message.source = _id;
		message.destination = id;
		message.payload = data;
		message.type = Message::Regular;

		report( "Thread " << _id << " sending message to " 
			<< message.destination << "." );
		assert( _group != 0 );
		_group->push( message );
	
	}
	
	Thread::Thread()
	{
		_id = _nextId++;
		_running = false;
		_group = new Group;
		_group->add( this );
		_thread = 0;
	}

	Thread::Thread( const Thread& t )
	{
		_id = _nextId++;
		_running = false;
		_group = new Group;
		_group->add( this );
		_thread = 0;
	}
	
	const Thread& Thread::operator=( const Thread& t )
	{
		assert( _running == false );
		assert( _group->size() == 1 );
		assert( t._group->size() == 1 );
		assert( t._running == false );
		return *this;
	}

	Thread::~Thread()
	{
		if( _running )
		{
			join();
		}

		assert( !_running );
		
		_group->remove( this );
		
		if( _group->empty() )
		{
			delete _group;
		}
	}

	void Thread::start()
	{
		assert( !_running );
		
		_running = true;
		
		report( "Thread " << _id << " starting." );

		_thread = new boost::thread( Thread::_launch, this );
	}

	void Thread::join()
	{
		assert( _running );
		
		report( "Thread " << _id << " joining." );
		
		_thread->join();
		delete _thread;
		_thread = 0;

		_running = false;
	}

	void Thread::associate( Thread* t )
	{
		assert( _group->size() == 1 || t->_group->size() == 1 );
		assert( id() != t->id() );
	
		if( _group->size() == 1 )
		{
			report( "Thread " << _id << " joining group with thread "
				<< t->id() << "." );
			_group->remove( this );
			delete _group;
			_group = t->_group;
			_group->add( this );
		}
		else
		{
			report( "Thread " << t->id() << " joining group with thread "
				<< id() << "." );
			t->_group->remove( t );
			delete t->_group;
			t->_group = _group;
			_group->add( t );
		}
	}

	void Thread::remove()
	{
		report( "Thread " << _id << " leaving group." );
		assert( !_group->empty() );
		
		if( _group->size() > 1 )
		{
			report( " Thread " << _id << " destroying group." );
		
			_group->remove( this );
			_group = new Group;
		}
	}

	void Thread::send( MessageData data )
	{
		report( "Thread " << _id 
			<< " sending message from controller to local thread." );
	
		Message message;
		message.source = THREAD_CONTROLLER_ID;
		message.destination = _id;
		message.payload = data;
		message.type = Message::Regular;
		
		_group->push( message );
	}

	bool Thread::test( bool block )
	{
		return _group->test( _id, block );
	}
	
	std::pair< Thread::Id, bool > Thread::testGroup( bool block, Id source )
	{
		std::pair< Id, bool > result;

		result.first = source;
		result.second = _group->test( result.first, block );

		assert( !block || result.second );
		
		return result;
	}

	Thread::Id Thread::id() const
	{
		return _id;
	}
	
	bool Thread::started() const
	{
		return _running;
	}
	
	bool Thread::killed() const
	{
		if(!started())
		{
			return true;
		}
		else
		{	
			#ifdef _WIN32
			DWORD code = 0;
			GetExitCodeThread(_thread->native_handle(), &code);
			return code != STILL_ACTIVE;
			#else
			return false;
			#endif
		}
	}
	
	Thread* Thread::find( Thread::Id id )
	{
		report( "Looking up thread with id " << id );
	
		if( id == _id )
		{
			return this;
		}
		return _group->find( id );
	}
}

////////////////////////////////////////////////////////////////////////////////
#endif

