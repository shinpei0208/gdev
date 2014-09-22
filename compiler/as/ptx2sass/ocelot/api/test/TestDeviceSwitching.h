/*! \file TestDeviceSwitching.h
	\date Saturday January 23, 2010
	\author Gregory Diamos
	\brief The header file for the TestDeviceSwitching class.
*/

#ifndef TEST_DEVICE_SWITCHING_H_INCLUDED
#define TEST_DEVICE_SWITCHING_H_INCLUDED

#include <hydrazine/interface/Test.h>

namespace test
{
	/*! \brief A unit test for the ability of the CUDA runtime to switch
		between devices and use multiple threads 
	
		Test Points:
			1) In a single threaded application, iterate across all devices
				launching the same simple kernel each time
			2) Test context migration support in ocelot.  Iterate across
				all devices, launching a simple kernel that updates shared
				variable.  Use the context switch mechanism to migrate
				the shared data.
			3) Test multi-threading support.  Launch one host thread to allocate
				memory, pass pointers to worker threads, each of which should
				launch independent kernels in parallel.
	*/
	class TestDeviceSwitching : public Test
	{
		private:
			/*! \brief Iterates over all devices launching one kernel each */
			bool testSwitching();
			/*! \brief Tests the ocelot context switch function */
			bool testContextSwitching();
			/*! \brief Test ocelot using multiple host threads */
			bool testThreads();
	
			/*! \brief Dispatch function fot the unit tests */
			bool doTest();
	
		public:
			/*! \brief The constructor sets the description */
			TestDeviceSwitching();
	};
}

int main(int argc, char** argv);

#endif

