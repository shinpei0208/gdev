/*!
	\file TestArgumentParser.h
	\author Gregory Diamos
	\date July 4 2008
	\brief Contains the class prototype for the Test infertace
*/


#ifndef TEST_ARGUMENT_PARSER_H_INCLUDED
#define TEST_ARGUMENT_PARSER_H_INCLUDED

#define TOTAL_TESTS TEST_INTS+TEST_DOUBLES+TEST_STRINGS + TEST_BOOLS
#define TOTAL_ARGS ((TEST_INTS+TEST_DOUBLES+TEST_STRINGS) * 2 + TEST_BOOLS)

#include <hydrazine/interface/Test.h>

namespace test
{

	class TestArgumentParser : public Test
	{
		private:
			bool doTest( );
	
		public:
			unsigned int intCount;
			unsigned int doubleCount;
			unsigned int stringCount;
			unsigned int boolCount;
	
		public:	
			TestArgumentParser();
	
	};
	
}

int main( int argc, char** argv );

#endif

