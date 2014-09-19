/*!
	\file TestBTree.h
	\date Monday May 18, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the TestBTree class.
*/

#ifndef TEST_B_TREE_H_INCLUDED
#define TEST_B_TREE_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <hydrazine/implementation/BTree.h>
#include <hydrazine/implementation/MmapAllocator.h>
#include <vector>
#include <map>

#define PAGE_SIZE 4
//#define ALLOCATOR hydrazine::MmapAllocator< std::pair< const unsigned int, unsigned int > >
#define ALLOCATOR std::allocator< std::pair< const unsigned int, unsigned int > >

namespace test
{

		/*!
		\brief A unit test for a BTree data structure implementing
			the STL map interface.
		
		Test Points:
		
			1) Randomly insert and remove elements from a std::map and a BTree
				assert that the final versions have the exact same elements
				stored in the same order.
				
			2) Add elements and then clear the BTree.  Assert that there are
				no elements after the clear and that the correct number is 
				reported by size after each insertion.
			
			3) Test iterating through the BTree.
			
			4) Test each of the comparison operators.
			
			5) Test searching functions.
			
			6) Test swapping with another map
			
			7) Test all of the insert functions.
			
			8) Test all of the erase functions.
			
			9) Test assignment and copy constructors.
			
			10) Do not run any tests, simply add a sequence to the localMap 
				and write it out to graph viz files after each operaton.

	*/
	class TestBTree : public Test
	{
		public:
			typedef hydrazine::BTree< unsigned int, unsigned int, 
				std::less<unsigned int>, ALLOCATOR, 
				PAGE_SIZE > Tree;
			typedef std::vector< unsigned int > Vector;
			typedef std::map< unsigned int, unsigned int > Map;
		
		private:
			void _init( Vector& v );
		
		private:
			bool testRandom();
			bool testClear();
			bool testIteration();
			bool testComparisons();
			bool testSearching();
			bool testSwap();
			bool testInsert();
			bool testErase();
			bool testCopy();
			void doBenchmark();
			bool doTest();
		
		public:
			bool benchmark;
			unsigned int elements;
			unsigned int iterations;
			std::string path;
		
		public:
			TestBTree();
		
	};

}

int main( int argc, char** argv );

#endif

