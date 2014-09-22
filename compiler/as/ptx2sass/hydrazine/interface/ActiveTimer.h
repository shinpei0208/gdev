/*!

	\file ActiveTimer.h

	\date Wednesday September 24, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The header file for the ActiveTimer class.

*/

#ifndef ACTIVE_TIMER_H_INCLUDED
#define ACTIVE_TIMER_H_INCLUDED

#include <boost/thread.hpp>
#include <hydrazine/interface/Timer.h>
#include <map>

#ifndef ACTIVE_TIMER_STACK_SIZE
#define ACTIVE_TIMER_STACK_SIZE 1024
#endif

namespace hydrazine
{
	class ActiveTimer
	{
		private:
			class SharedData
			{
				public:
					typedef std::map<Timer::Second, ActiveTimer*> TimerMap;
			
				public:
					TimerMap timers;
					boost::thread* thread;
					boost::mutex mutex;
					boost::condition_variable condition;
					Timer timer;
					unsigned int connections;
					bool alive;

				public:
					void next();
				
				public:
					static void* run(void* argument);
			
				public:
					SharedData();
					~SharedData();
			
			};
			
			friend class SharedData;
	
		private:
			static SharedData _sharedData;
			
		private:
			boost::condition_variable _condition;
			
		protected:
			bool _done;

		private:
			virtual void fired() = 0;
	
		public:
			ActiveTimer();
			~ActiveTimer();
			
			ActiveTimer(const ActiveTimer& timer);
			const ActiveTimer& operator=(const ActiveTimer& timer);
			
			void start(Timer::Second seconds);
			void wait();			
	
	};

}

#endif

