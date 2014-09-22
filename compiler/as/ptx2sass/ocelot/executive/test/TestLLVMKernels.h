/*!
	\file TestLLVMKernels.h
	\date Friday September 4, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the TestLLVMKernels unit test.
*/

#ifndef TEST_LLVM_KERNELS_H_INCLUDED
#define TEST_LLVM_KERNELS_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <ocelot/executive/interface/LLVMExecutableKernel.h>
#include <ocelot/ir/interface/Module.h>

namespace test
{
	/*!
		\brief A unit test for the LLVM executive runtime.
		
		Test Points:
			1) Execute a kernel with a loop.
			2) Execute a matrix multiply kernel.
	*/
	class TestLLVMKernels : public Test
	{
		private:
			executive::LLVMExecutableKernel* _loopingKernel;
			executive::LLVMExecutableKernel* _matrixMultiplyKernel;
			ir::Module _module;		

		private:
			bool _loadKernels();
		
			bool testLooping();
			bool testMatrixMultiply();
			
			bool doTest();
		
		public:
			TestLLVMKernels();
			~TestLLVMKernels();
			
		public:
			std::string kernelFile;
	};
}

int main( int argc, char** argv );

#endif

