/*! \file TestOptimizations.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday May 1, 2012
	\brief The header file for the TestPTXAssembly class.
*/

#pragma once

#include <hydrazine/interface/Test.h>

namespace test
{

/*! \brief Define a test harness for running PTX kernels
	through various optimization passes.
*/
class TestOptimizations : public Test
{
public:
	/*! \brief Total amount of time to spend on tests in seconds */
	hydrazine::Timer::Second timeLimit;
	/*! \brief Only enumerate the tests, do not run them */
	bool enumerate;
	/*! \brief print out status information as the test is running */
	bool veryVerbose;
	/*! \brief search path for PTX files */
	std::string path;
	/*! \brief The set of optimizations to run */
	std::string optimizations;

public:
	/*! \brief Constructor  */
	TestOptimizations();
	
public:
	/*! \brief Run the current set of tests, abort on the first error */
	bool doTest();
	/*! \brief Run the specified test, report passed/failed */
	bool runTest(const std::string& test);
};

}


