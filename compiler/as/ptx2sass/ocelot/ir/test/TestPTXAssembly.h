/*! \file TestPTXAssembly.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday May 11, 2010
	\brief The header file for the TestPTXAssembly class.
*/

#ifndef TEST_PTX_ASSEMBLY_H_INCLUDED
#define TEST_PTX_ASSEMBLY_H_INCLUDED

#include <ocelot/ir/interface/Instruction.h>
#include <hydrazine/interface/Test.h>
#include <vector>

namespace test
{
	/*! \brief The idea here is to define a test harness for a large number 
		of PTX unit tests.
	*/
	class TestPTXAssembly : public Test
	{
		public:
			/*! \brief Possible data types */
			enum DataType
			{
				I8 = 0,
				I16 = 1,
				I32 = 2,
				I64 = 3,
				FP32 = 4,
				FP64 = 5
			};
		
			/*! \brief A vector of data types */
			typedef std::vector<DataType> TypeVector;

			/*! \brief A pointer to a reference function */
			typedef void (*ReferenceFunction)(void* output, void* input);
			
			/*! \brief A pointer to an input generator function */
			typedef char* (*GeneratorFunction)(MersenneTwister& generator);
			
			/*! \brief A class for representing a single test */
			class TestHandle
			{
				public:
					std::string name;
					ReferenceFunction reference;
					GeneratorFunction generator;
					std::string ptx;
					TypeVector inputTypes;
					TypeVector outputTypes;
					unsigned int threads;
					unsigned int ctas;
					int epsilon;
			};
			
			/*! \brief A list of tests to perform */
			typedef std::vector<TestHandle> TestVector;
	
		private:
			/*! \brief The list of tests to run */
			TestVector _tests;
			/*! \brief The number of tolerable failures */
			unsigned int _tolerableFailures;
			/*! \brief The PTX device count */
			unsigned int _devices;
			
		public:
			/*! \brief Get the size of a data type in bytes */
			static unsigned int bytes(DataType t);
		
		private:
			/*! \brief Perform a single unit test */
			bool _doOneTest(const TestHandle& test, unsigned int seed);
			/*! \brief Load the tests */
			void _loadTests(ir::Instruction::Architecture ISA = ir::Instruction::Emulated);
			
			static void _writeArguments(std::ostream &out, const TypeVector &, char *);
		public:
			/*! \brief Print out the ptx of each test as it is added */
			bool print;
			/*! \brief Only add tests that match this regular expression */
			std::string regularExpression;
			/*! \brief Total amount of time to spend on tests in seconds */
			hydrazine::Timer::Second timeLimit;
			/*! \brief Only enumerate the tests, do not run them */
			bool enumerate;
			/*! \brief print out status information as the test is running */
			bool veryVerbose;
		
		public:
			/*! \brief Constructor */
			TestPTXAssembly(hydrazine::Timer::Second limit = 10.0, 
				unsigned int tolerableFailures = 0);
			
		public:
			/*! \brief Add a test, ptx function name should be 'test' */
			void add(const std::string& name,
				ReferenceFunction function, const std::string& ptx, 
				const TypeVector& out, const TypeVector& in, 
				GeneratorFunction gen, unsigned int threads, unsigned int ctas,
				int epsilon = 0);
				
			/*! \brief Run the current set of tests, abort on the first error */
			bool doTest();
	};
}

int main(int argc, char** argv);

#endif

