/*! \file   TestExternalFunctions.h
	\date   Thursday April 7, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the TestExternalFunctions class.
*/

#ifndef TEST_EXTERNAL_FUNCTIONS_H_INCLUDED
#define TEST_EXTERNAL_FUNCTIONS_H_INCLUDED

// Hydrazine Includes
#include <hydrazine/interface/Test.h>

namespace test
{

/*! \brief A unit test for calling an external host function from PTX */
class TestExternalFunctions : public Test
{
public:
	/*! \brief The constructor sets the description and name */
	TestExternalFunctions();

public:
	bool testMallocFree();
	bool testPrintf();
	bool testUserFunction();
	bool testMismatchingTypes();

public:
	/*! \brief The entry point of the test */
	bool doTest();
};

}

int main(int argc, char** argv);

#endif

