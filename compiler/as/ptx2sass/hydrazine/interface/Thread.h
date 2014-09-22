/*! \file Thread.h
	\author Gregory Diamos
	\date 4/10/2008
	\brief 	The header file for the thread class
*/

#ifndef HYDRAZINE_THREAD_H_INCLUDED
#define HYDRAZINE_THREAD_H_INCLUDED

#include <boost/thread.hpp>
#include <hydrazine/interface/SystemCompatibility.h>

#include <list>
#include <cassert>

#define THREAD_CONTROLLER_ID 0
#define THREAD_START_ID ( THREAD_CONTROLLER_ID + 1 )
#define THREAD_ANY_ID 0xffffffff

namespace hydrazine
{

	/*!
		\brief A wrapper class around pthreads
		
		Essentially this is just a front end interface in the main process
		and a thread asociated with it, running in the background.  This
		class provides an api for communication between the main process
		and the associated thread via message passing.  The message passing 
		model assumes that the main process owns sections of memory.  The act of 
		sending a message from the main process to the thread transfers 
		ownership a specified section of memory from the main process to the 
		thread.  Threads can receive these memory segments, do some processing 
		on them, and possibly pass ownership back to the main process via their 
		own version of the send function.  All sends should be implemented via 
		mutexes, condition variables, and should pass pointers rather than
		doing copies.  
		
		The point here is to create a clean programming model for dealing with
		parallelism.  Each thread owns sections of memory and must use
		sends and recieves to transfer ownership.  No need to ever worry about
		aquiring locks for dealing with shared data.  It's up to the programmer
		to ensure that the implementation scales, but mutexes won't be the 
		bottleneck.
		
		Remimplementing this class requires
		the remiplementation of the execute function which corresponds to the 
		function that is executed by the spawned thread when it is created
		from the main process.
		
		Threads are referred to by id and many threads can be grouped together.
		Once this has been done, messages can be sent from any of the
		thread objects in the main process to any of the threads in the group.
		
		Similarly, once threads have been grouped together, the entire group
		can be tested to see if there are any messages from any of the threads 
		that can be received.
	*/
	class Thread
	{
		public:
			/*! \brief A type for thread ids */
			typedef unsigned int Id;
	
			/*! \brief A type for a message's payload data */
			typedef void* MessageData;
	
		private:
			class Message
			{
				public:
					enum Type
					{
						Invalid,
						Regular
					};
			
				public:
					MessageData payload;
					Id source;
					Id destination;
					Type type;
			};
	
			class Queue
			{
				private:
					friend class Group;
			
				private:
					typedef std::list< Message > MessageQueue;
				
				private:
					boost::condition_variable _condition;
					boost::mutex _mutex;
			
				private:
					MessageQueue _queue;
			
				public:
					Queue();
					~Queue();
					
					void push( const Message& );
					Message pull( Id );
					bool test( Id&, bool block = false );
			
			};

			class Group
			{
				private:
					boost::mutex _mutex;
			
				public:
					typedef std::unordered_map< Id, Thread* > ThreadMap;

				private:
					ThreadMap _threads;
					Queue _controllerQueue;
				
				public:
					Group();
					~Group();
					
					void add( Thread* );
					void remove( Thread* );
					Thread* find( Id );
					
					void push( const Message& );
					Message pull( Id );
					bool test( Id&, bool );
			
					bool empty() const;
					unsigned int size() const;
			};
	
		private:
			/*! \brief The next id */
			static Id _nextId;

		private:
			/*! \brief Pthread create needs a static function to launch 
					the thread with
				
				\return Void pointer required by pthreads.
				
				\param argument The argument passed to the thread.
			
			*/
			static void* _launch( void* argument );

			/*! \brief Compare two ids */
			static bool _compare( Id, Id );

		private:
			/*! \brief Is the thread running */
			bool _running;

			/*! \brief Queue of messages to be delivered to the thread */
			Queue _threadQueue;
			
			/*! \brief Group of threads
				
				Should be set to 0 if threads are not grouped
			*/
			Group* _group;
			
			/*! \brief The thread handle */
			boost::thread* _thread;

			/*! \brief The thread id */
			Id _id;
			
		protected:
		
			/*! \brief This is the function that is executed in a separate 
					thread when the run command is sent.
			*/
			virtual void execute() = 0;
	
		protected:
	
			/*!\brief Test to see if there are any messages that can be 
					received
			
				\return True if there is a message that can be received
				
				\param The id to receive from
				
				\param any Receive a message from any id
				
			*/
			bool threadTest( Id id = THREAD_ANY_ID );

			/*! \brief Receive a message in this thread.
			
				This method will block until the message is received
				
				\return the id of the message source
			
				\param The id to receive from
							
			*/
			template<class T>
			Id threadReceive( T*& message, Id id = THREAD_ANY_ID );
						
			/*! \brief Send a message to the controller thread
			
				This method will not block even if the message was not received 
					by the controller.
			
				\param id The thread id to send the message to.
			
			*/
			void threadSend( MessageData message, 
				Id id = THREAD_CONTROLLER_ID );
			
			/*! \brief All associated threads will block here until all other 
					threads have hit the barrier */
			void barrier();					
			
		public:
		
			/*! \brief Constructor */
			Thread();
			
			/*! \brief Copy constructor */
			Thread( const Thread& );
			
			/*! \brief Assignment operator */
			const Thread& operator=( const Thread& );
			
			/*! \brief Destructor */
			virtual ~Thread();
		
			/*! \brief Start the thread */
			void start();
			
			/*! \brief Block until the thread returns */
			void join();

			/*! \brief Associate this thread with another thread
				
				This is used for collective operations
			
				\param The thread to associate with
			*/
			void associate( Thread* t );
		
			/*! \brief Remove this thread from any groups that it is 
					associated with
			*/
			void remove();
		
			/*! \brief Send a message to this thread
			
				This method will not block until the message is received by the 
					thread
			
				\param The message being sent
			*/
			void send( MessageData message );
			
			/*! \brief Test to see if there are any messages that can be 
					received
			
				\param Block until there is a message
			
				\return True if there is a message that can be received
			*/
			bool test( bool block = false );
			
			/*! \brief Test to see if there are any messages in any of the 
					queues of the threads in the current group.
				
				\param id Test for messages send by a specific thread
				
				\param block Should the function block until it returns true?
				
				\return True if there is a message in the receive queue of any
				thread in the group along with it's id
			
			*/
			std::pair< Id, bool > testGroup( bool block = false, 
				Id source = THREAD_ANY_ID );
			
			/*! \brief Receive a message from this thread.
			
				This method will block until the message is received
				
				\param message A pointer to the message being received

			*/
			template<class T>
			void receive( T*& message );
			
			/*! \brief Get the id of this thread.
			
				\return The id of this thread.
			*/
			Id id() const;
			
			/*! \brief Is the thread running
			
				\return true if it is running
			*/
			bool started() const;
			
			/*! \brief Was the thread killed?
			
				\return true if it has been killed
			*/
			bool killed() const;
			
			/*! \brief Get a pointer to a thread in the group associated with a 
				specific id.
				
				\param id The id of the thread in the group to look up.
				
				\return the pointer to the correct thread.
				
				should return 0 if thread does not exist
			*/
			Thread* find( Id id );
	
	};

}

#include "Thread.hpp"

#endif

