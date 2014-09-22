/*! \file   TestExternalFunctions.cpp
	\date   Thursday April 7, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the TestExternalFunctions class.
*/

#ifndef TEST_EXTERNAL_FUNCTIONS_CPP_INCLUDED
#define TEST_EXTERNAL_FUNCTIONS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/api/test/TestExternalFunctions.h>
#include <ocelot/api/interface/ocelot.h>

#include <ocelot/cuda/interface/cuda_runtime.h>

#include <configure.h>

// Standard Library Includes
#include <fstream>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

namespace test
{

TestExternalFunctions::TestExternalFunctions()
{
	name = "TestExternalFunctions";
	
	description = "A unit test for calling an external host function from PTX\n"
		"\n"
		"\t1) The first test calls malloc/free from a PTX kernel.\n"
		"\t2) The second test calls printf from a PTX kernel.\n"
		"\t3) The last test calls a user defined function from a PTX kernel.\n";
}

bool TestExternalFunctions::testMallocFree()
{
	std::string ptx = ".version 2.3\n"
		".address_size 64\n"
		"\n"
		".extern .func (.param .u64 pointer) malloc (.param .u64 bytes)\n"
		".extern .func free (.param .u64 pointer)\n"
		"\n"
		".entry kernel(.param .u64 result) {\n"
		"\t.reg .u64 %r<10>;\n"
		"\t.param .u64 returned;\n"
		"\t.param .u64 value;\n"
		"\t.param .u64 freeValue;\n"
		"\t\n"
		"\tmov.u64 %r0, 8;\n"
		"\tst.param.u64 [value], %r0;\n"
		"\tcall.uni (returned), malloc, (value);\n"
		"\tld.param.u64 %r1, [returned];\n"
		"\tst.global.u64 [%r1], %r0;\n"
		"\tld.global.u64 %r2, [%r1];\n"
		"\tld.param.u64 %r3, [result];\n"
		"\tst.global.u64 [%r3], %r2;\n"
		"\tst.param.u64 [freeValue], %r1;\n"
		"\tcall.uni free, (freeValue);\n"
		"\texit;\n"
		"}\n";

	std::stringstream stream(ptx);
	ocelot::registerPTXModule(stream, "someModule");
	
	long long unsigned int* sizeOfData = 0;
	cudaMalloc((void**)&sizeOfData, sizeof(long long unsigned int));
	
	cudaSetupArgument(&sizeOfData, sizeof(long long unsigned int), 0);
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);
	ocelot::launch("someModule", "kernel");
	
	long long unsigned int result = 0;
	
	cudaMemcpy(&result, sizeOfData, sizeof(long long unsigned int),
		cudaMemcpyDeviceToHost);
	
	cudaFree(sizeOfData);
	ocelot::unregisterModule("someModule");
	
	return result == sizeof(long long unsigned int);
	
}

bool TestExternalFunctions::testPrintf()
{
	std::string pointerType;
	
	if(sizeof(size_t) == 4)
	{
		pointerType = ".u32";
	}
	else
	{
		pointerType = ".u64";	
	}

	std::string ptx = ".version 2.3\n"
		"\n"
		".extern .func (.param .s32 return) vprintf (.param "
		+ pointerType + " string, .param " + pointerType + " parameters)\n"
		"\n"
		".global .align 1 .b8 message[15]"
		" = {0x48,0x65,0x6c,0x6c,0x6f,0x20,0x43,0x55,0x44,"
		" 0x41,0x20,0x25,0x64,0xa,0x0};\n"
		"\n"
		".entry kernel() {\n"
		"\t\t.reg " + pointerType + " %r<5>;\n"
		"\t\t.local .align 8 .b8 parameters[8];\n"
		"\t\t.param " + pointerType + " string;\n"
		"\t\t.param " + pointerType + " parameters_pointer;\n"
		"\t$begin:\n"
		"\t\tmov" + pointerType + "        %r1,                  2;\n"
		"\t\tst.local" + pointerType + "   [parameters+0],       %r1;\n"
		"\t\tcvta.global" + pointerType + " %r2,                  message;\n"
		"\t\tst.param" + pointerType + "   [string],             %r2;\n"
		"\t\tcvta.local" + pointerType + "  %r3,                  parameters;\n"
		"\t\tst.param" + pointerType + "   [parameters_pointer], %r3;\n"
		"\t\tcall.uni (_),      vprintf, (string, parameters_pointer);\n"
		"\t\texit;\n"
		"\t$end:\n"
		"}\n";

	std::stringstream stream(ptx);

	ocelot::registerPTXModule(stream, "someModule");
	
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);

	ocelot::launch("someModule", "kernel");

	ocelot::unregisterModule("someModule");
	
	return true;
}

static void randomHostFunction(long long unsigned int address)
{
	(*(int*)address) = 0xfeedcaa7;
}

bool TestExternalFunctions::testUserFunction()
{
	std::string ptx = ".version 2.3\n"
		".address_size 64\n"
		"\n"
		".extern .func hostFunction (.param .u64 bytes)\n"
		"\n"
		".entry kernel(.param .u64 result) {\n"
		"\t.reg .u64 %r<10>;\n"
		"\t.param .u64 value;\n"
		"\t\n"
		"\tld.param.u64 %r0, [result];\n"
		"\tst.param.u64 [value], %r0;\n"
		"\tcall.uni hostFunction, (value);\n"
		"\texit;\n"
		"}\n";

	std::stringstream stream(ptx);
	ocelot::registerExternalFunction("hostFunction",
		(void*)(randomHostFunction));

	ocelot::registerPTXModule(stream, "someModule");
	
	int data = 0;
	long long unsigned int address = (long long unsigned int)(&data);
	
	cudaSetupArgument(&address, sizeof(long long unsigned int), 0);
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);
	ocelot::launch("someModule", "kernel");
	
	ocelot::unregisterModule("someModule");
	ocelot::removeExternalFunction("hostFunction");
	
	return data == (int)0xfeedcaa7;
}

static void randomHostFunction2(long long unsigned int address, unsigned int d)
{
	(*(unsigned int*)address) = d;
}

bool TestExternalFunctions::testMismatchingTypes()
{
	std::string ptx = ".version 2.3\n"
		".address_size 64\n"
		"\n"
		".extern .func hostFunction (.param .u64 bytes, .param .u32 data)\n"
		"\n"
		".entry kernel(.param .u64 result, .param .u32 result2) {\n"
		"\t.reg .u64 %r<10>;\n"
		"\t.param .u64 value0;\n"
		"\t.param .u32 value1;\n"
		"\t\n"
		"\tld.param.u64 %r0, [result];\n"
		"\tld.param.u32 %r1, [result2];\n"
		"\tst.param.u64 [value0], %r0;\n"
		"\tst.param.u32 [value1], %r1;\n"
		"\tcall.uni hostFunction, (value0, value1);\n"
		"\texit;\n"
		"}\n";

	std::stringstream stream(ptx);
	ocelot::registerExternalFunction("hostFunction",
		(void*)(randomHostFunction2));

	ocelot::registerPTXModule(stream, "someModule");
	
	unsigned int data = 0;
	long long unsigned int address = (long long unsigned int)(&data);
	unsigned int value = 0xcaa7f00d;
	
	cudaSetupArgument(&address, sizeof(long long unsigned int), 0);
	cudaSetupArgument(&value, sizeof(unsigned int),
		sizeof(long long unsigned int));
	cudaConfigureCall(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);
	ocelot::launch("someModule", "kernel");
	
	ocelot::unregisterModule("someModule");
	ocelot::removeExternalFunction("hostFunction");

	if(data != 0xcaa7f00d)
	{
		status << "TestMismatchingTypes failed, "
			"expecting 0xcaa7f00d, got " << std::hex << data 
			<< std::dec << "\n";
	}
	
	return data == 0xcaa7f00d;
}

bool TestExternalFunctions::doTest()
{
	#if HAVE_LLVM
	return testMallocFree() 
		&& testPrintf() 
		&& testUserFunction()
		&& testMismatchingTypes();
	#else
	return true;
	#endif
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	test::TestExternalFunctions test;
	parser.description(test.testDescription());

	parser.parse( "-s", "--seed", test.seed, 0, 
		"Random number generator seed, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out information after the test has finished." );
	parser.parse();
	
	test.test();
		
	return test.passed();	
}

#endif

