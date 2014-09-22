/*!

	\file TestLexer.h
	
	\date Saturday January 17, 2009
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The header file for the TestLexer class

*/

#ifndef TEST_LEXER_H_INCLUDED
#define TEST_LEXER_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <deque>

namespace test
{

	/*!
	
		\brief Tests for the PTX lexer
		
		Test Point 1: Scan a PTX file and write out a temp stream, scan the 
			stream again and make sure that the two sets of tokens match
	
	*/
	class TestLexer : public Test
	{

		public:
		
			typedef std::deque< std::string > StringVector;
	
		private:
		
			std::string ptxFile;
		
		private:
		
			StringVector _getFileNames() const;
		
			bool _testScan();
		
			bool doTest();
		
		public:
		
			TestLexer();
		
		public:
		
			std::string input;
			bool recursive;
	
	};

}

int main( int argc, char** argv );

#endif

