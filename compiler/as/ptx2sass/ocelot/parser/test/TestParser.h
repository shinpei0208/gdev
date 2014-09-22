/*!
	\file TestParser.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Monday January 19, 2009
	\brief The header file for the TestParser class
*/

#ifndef TEST_PARSER_H_INCLUDED
#define TEST_PARSER_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <deque>

namespace test
{

	/*!
		\brief A test for the PTXParser class

		Test Points:
			1) Load a PTX file and run it through the parser generating a 
				module.  Write the module to an intermediate stream.  Parse the
				stream again generating a new module, compare both to make sure
				that they match.
	*/
	class TestParser : public Test
	{
		public:
			typedef std::deque< std::string > StringVector;
	
		private:
			std::string ptxFile;
		
		private:
			StringVector _getFileNames() const;
			bool _testParse();
			bool doTest();
		
		public:
			TestParser();
		
		public:
			std::string input;
			/*! \brief Total amount of time to spend on tests in seconds */
			hydrazine::Timer::Second timeLimit;
			bool recursive;
			bool output;

	};

}

int main( int arch, char** argv );

#endif

