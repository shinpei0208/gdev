/*!
	\file TestDataflowGraph.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday July 7, 2009
	\brief The header file for the TestDataflowGraph class.
*/

#ifndef TEST_DATAFLOW_GRAPH_H_INCLUDED
#define TEST_DATAFLOW_GRAPH_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <deque>

namespace test
{
	/*!
		\brief A test for the DataflowGraph class.
		
		Test Points:
			1) Generic: load PTX files, convert them into dataflow graphs,
				verify that all live ranges spanning blocks are consistent.
			2) SSA: convert to ssa form, verify that no register is declared
				more than once.
			3) reverse SSA: convert to ssa then out of ssa, verify that all
				live ranges spanning blocks are consistent.
	*/
	class TestDataflowGraph : public Test
	{
		private:
			/*! \brief A vector to hold PTX files to process */
			typedef std::deque< std::string > StringVector;
		
		private:
			/*! \brief A list of all ptx files to test */
			StringVector _files;
		
		private:
			bool _verify( const analysis::DataflowGraph& graph );
			bool _verifySsa( const analysis::DataflowGraph& graph );
		
		private:
			/*! \brief Scan the input directory for all ptx files */
			void _getFileNames();
			
			/*! \brief Test 1 */
			bool _testGeneric();
			
			/*! \brief test 2 */
			bool _testSsa();
			
			/*! \brief test 3 */
			bool _testReverseSsa();
			
			/*! \brief Perform all of the tests. */
			bool doTest();
		
		public:
			/*! \brief The relative path to search for PTX files */
			std::string base;
			/*! \brief Total amount of time to spend on tests in seconds */
			hydrazine::Timer::Second timeLimit;
		
		public:
			TestDataflowGraph();
			
	};
}

int main( int argc, char** argv );

#endif

