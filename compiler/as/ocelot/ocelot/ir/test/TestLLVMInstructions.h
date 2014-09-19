/*!
	\file TestLLVMInstructions.h
	\date Monday July 27, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the TestLLVMInstructions class.
*/

#ifndef TEST_LLVM_INSTRUCTIONS_H_INCLUDED
#define TEST_LLVM_INSTRUCTIONS_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <ocelot/ir/interface/LLVMInstruction.h>

namespace test
{
	/*! \brief A test for the assembly code generation and automatic verfication
		of individual LLVM instructions.
		
		Test Points:
			1) For each instruction, generate several assembly strings using
				the instruction's toString method, make sure that these pass
				the valid() check, compare to references from the LLVM manual.
		
	*/ 
	class TestLLVMInstructions : public Test
	{
		public:
			typedef ir::LLVMInstruction Instruction;
			typedef ir::LLVMInstruction::Type Type;
			typedef ir::LLVMInstruction::Operand Operand;
	
		private:
			bool check( const ir::Instruction& i, 
				const std::string& reference );
	
		private:
			bool testAdd();
			bool testAlloca();
			bool testAnd();
			bool testAshr();
			bool testBitcast();
			bool testBr();
			bool testCall();
			bool testExtractelement();
			bool testExtractvalue();
			bool testFadd();
			bool testFcmp();
			bool testFdiv();
			bool testFmul();
			bool testFpext();
			bool testFptosi();
			bool testFptoui();
			bool testFptrunc();
			bool testFree();
			bool testFrem();
			bool testFsub();
			bool testGetelementptr();
			bool testIcmp();
			bool testInsertelement();
			bool testInsertvalue();
			bool testInttoptr();
			bool testInvoke();
			bool testLoad();
			bool testLshr();
			bool testMalloc();
			bool testMul();
			bool testOr();
			bool testPhi();
			bool testPtrtoint();
			bool testRet();
			bool testSdiv();
			bool testSelect();
			bool testSext();
			bool testShl();
			bool testShufflevector();
			bool testSitofp();
			bool testSrem();
			bool testStore();
			bool testSub();
			bool testSwitch();
			bool testTrunc();
			bool testUdiv();
			bool testUitofp();
			bool testUnreachable();
			bool testUnwind();
			bool testUrem();
			bool testVaArg();
			bool testXor();
			bool testZext();
			
			bool doTest();
		
		public:
			TestLLVMInstructions();
	};
}

int main( int argc, char** argv );

#endif

