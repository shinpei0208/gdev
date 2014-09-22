/*!
	\file TestPTXToLLVMTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Sunday August 16, 2009
	\brief A test for the PTXToLLVMTranslator class.
*/

#ifndef TEST_PTX_TO_LLVM_TRANSLATOR_H_INCLUDED
#define TEST_PTX_TO_LLVM_TRANSLATOR_H_INCLUDED

#include <hydrazine/interface/Test.h>

namespace test
{
	
	/*! \brief This is a basic test that just tries to get through 
			a translation successfully of as many PTX programs as possible
			
		Test Points:
			1) Scan for all PTX files in a directory, try to translate them. 
	*/
	class TestPTXToLLVMTranslator : public Test
	{
		public:
			typedef std::deque< std::string > StringVector;
	
		private:
			std::string ptxFile;
		
		private:
			StringVector _getFileNames() const;
			bool _testTranslate();
			bool doTest();
		
		public:
			TestPTXToLLVMTranslator();
		
		public:
			std::string input;
			/*! \brief Total amount of time to spend on tests in seconds */
			hydrazine::Timer::Second timeLimit;
			bool recursive;
			bool output;
	};

}

int main( int argc, char** argv );

#endif

